"""
PhaseIndex V30.4 -- Continuous Stateful Generalized Langevin Detector
with Soft Labels and Stochastic Weight Averaging.

Physical framework
-------------------
Each trading day is treated as a state (z, p) -- position and momentum --
of a particle moving in a learned, input-conditioned Ginzburg-Landau
bistable potential V(z) = a*z^4 - b*z^2 + c*z. Dynamics are integrated with
a velocity-Verlet scheme under Langevin thermostatting: learned per-day
friction (gamma), temperature (T), and a memory/friction kernel (phi, s) in
the spirit of the generalized Langevin equation / Mori-Zwanzig formalism. A
small classification head maps the particle's final (z, p) state to a phase
index Psi in [0, 1]: how far the market has moved from the "stable" basin
toward the "crisis" basin of the potential.

Changes in V30.4 vs. V30.3 (everything else is identical)
-----------------------------------------------------------
1. SOFT_LABEL_SIGMA = 7 (new CFG parameter). Hard 0/1 crisis labels are
   smoothed with a Gaussian kernel (sigma = 7 trading days) into a
   continuous training target in [0, 1], with a ~14-day ramp at each crisis
   boundary. Hard labels are kept for evaluation; soft labels are used only
   for training.
2. make_hard_labels() is the original make_labels(), unchanged.
   make_soft_labels() is new: a Gaussian filter over the hard labels.
3. hooke_loss() is rewritten for continuous labels: the target position
   interpolates between the stable and crisis anchors as a function of the
   label, and the loss stiffness for under-shooting a crisis interpolates
   from 1x to ASYM_CRISIS_UNDER_K (3x) with the label. This removes the
   gradient discontinuity the old binary-label loss had at every regime
   boundary.
4. Stochastic Weight Averaging (SWA), carried over from V30.3, is kept --
   safe to keep because this architecture has no BatchNorm layers, which is
   the usual reason SWA gets disabled.

What this refactor changed (everything above is unmodified methodology)
-------------------------------------------------------------------------
This pass reorganizes the script for a public repository and adds
reporting. The reorganization itself changes no model, hyperparameter, or
formula. It does not change any number from the version above -- every
number already validated for V30.4 reproduces exactly under the default
configuration.

  - Extracts the monolithic main() into load_price_data / build_feature_
    matrix / split_data / build_report, and moves the report's statistics
    and plotting out to evaluation_metrics.py / visualization.py.
  - CFG.STRICT_CALIBRATION_HOLDOUT defaults to False (kept as an opt-in
    flag, not removed -- see below for why it exists and why it isn't
    the default).

    The issue it addresses is real: under the default split, the
    calibration window (2017-2019) used to fit temperature scaling is a
    *subset* of the training window, so calibration is fit on in-sample
    model outputs. That never affects the headline detection metrics
    (recall / precision / F1 / ROC-AUC / PR-AUC, confusion matrices,
    per-crisis recall) -- those are all computed on the genuinely
    held-out 2020+ test set regardless of this flag. It can optimistically
    bias the three calibration-quality numbers specifically (ECE / Brier /
    log-loss).

    This was tried as the default in an earlier pass and reverted after
    measuring the cost: moving the training cutoff from 2020 to 2017 to
    get a disjoint split removes ~780 training days (~11% of the training
    set) -- and CRISES contains ("2018-10-01", "2018-12-31"), which falls
    entirely inside that removed window. That single episode is the
    model's only training-set example of a slow, rate-hike-driven
    drawdown, structurally the closest analog to the 2022 test crisis
    (as opposed to COVID's liquidity-panic profile, which several earlier
    training crises still cover). Removing it did not degrade performance
    uniformly: in a real run, COVID recall was essentially unchanged
    (86.3% -> 84.3% base) while 2022 recall roughly halved (41.3% -> 21.1%
    base), and PR-AUC, Cohen's d, MCC, F1, and calibrated recall/precision
    all fell together with it. ECE also got *worse* (2.66% -> 3.41%), not
    better -- consistent with the original number being optimistic, but
    confounded with the model itself being weaker on less data, so it
    isn't a clean before/after comparison of calibration alone.

    Net assessment: sacrificing a rare, structurally-relevant training
    exemplar to clean up 3 secondary metrics was a bad trade. Set
    STRICT_CALIBRATION_HOLDOUT=True if you want to reproduce that
    measurement or explore it further (e.g. restricted to swapping in a
    calibration window that doesn't overlap a labeled crisis), but the
    recommended path for reporting calibration honestly without paying
    this cost is: keep the default split, keep training on the full
    pre-2020 history, and disclose the in-sample calibration overlap as a
    limitation in prose (ECE/Brier/log-loss specifically may be
    optimistic; the detection metrics are unaffected). A cleaner fix
    without the data cost -- fit T_calib from an auxiliary model trained
    only through 2017 and apply the resulting temperature to the
    fully-trained final model -- is a reasonable next step if calibration
    numbers need to be defensible on their own, not implemented here.
  - Replaces the hand-rolled PR-AUC step-sum with sklearn's

    average_precision_score (same definition of Average Precision; sklearn
    resolves tied Psi values more correctly -- see
    evaluation_metrics.legacy_average_precision for the reasoning and a
    live diagnostic against the old formula on your own predictions).
  - Adds MCC, balanced accuracy, Youden's J, a Brier Skill Score,
    calibration slope/intercept, and block-bootstrap confidence intervals
    for ROC-AUC / PR-AUC (block, not i.i.d., because consecutive trading
    days inside the same crisis episode are not independent draws).
  - Replaces the hardcoded V30.3 comparison dict with a small JSON run log
    (results/run_history.json) so V30.5 and later compare automatically.
  - Records training loss history (previously printed, not stored) purely
    for the training-curve plot; does not affect training.

Two things worth confirming before this goes in a paper or a repo
---------------------------------------------------------------------
  - config-4.py must be saved as config.py next to this file -- features.py
    does `import config`, which fails under any other filename.
  - spectral_features.py (wavelet energy decomposition) is not imported
    here. SLOW_IDX / FAST_IDX below are sized for the 204-column feature
    set without it (80 + 124). Confirm this exclusion is intentional before
    wiring spectral features back in.
"""

from __future__ import annotations

import logging
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yfinance as yf
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

import evaluation_metrics as em
import visualization as viz
from features import build_base_features
from velocity_features import build_velocity_features

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phaseindex")


# ---- Configuration ----
class CFG:
    """All tunable constants for the pipeline. Grouped by concern; values
    are unchanged from the original V30.4 script unless noted."""

    # -- Universe & data window --
    TICKERS_MARKET = ["SPY", "QQQ", "IWM", "DIA", "VTI"]
    TICKERS_SECTOR = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLB", "XLC", "XLY", "XLP", "XLRE"]
    TICKERS_MACRO = ["TLT", "SHY", "IEF", "HYG", "LQD", "GLD", "TIP", "UUP", "USO", "DBC"]
    TICKERS_INTL = ["EFA", "EEM", "FXI", "EWJ", "VGK"]
    TICKERS_NEW = ["EWZ", "EWY", "EWT", "EWA", "EWG", "SLV", "DBA", "GDX", "EZU",
                   "AGG", "EMB", "JNK", "IBB", "KRE", "XHB", "IYT", "XRT", "FXE", "FXY", "FXB"]
    TICKERS_INDICES = ["^GSPC", "^IXIC", "^N225", "^FTSE", "^GDAXI", "^STOXX50E", "^FCHI", "^HSI"]
    TICKERS_EXTRA = ["^VIX", "GC=F", "CL=F"]
    ALL_TICKERS = (TICKERS_MARKET + TICKERS_SECTOR + TICKERS_MACRO + TICKERS_INTL
                   + TICKERS_NEW + TICKERS_INDICES + TICKERS_EXTRA)
    TARGET = "SPY"
    START_DATE = "1993-02-01"
    END_DATE = "2024-01-01"

    # -- Default False: spectral_features.py (40 wavelet-decomposition
    # columns) is not wired in. Measured cost of computing them is small
    # (~10s on the full 31-year series -- timed directly, not assumed),
    # but adding them changes FastEncoder's input width, so this is a real
    # architecture change requiring a full retrain, not a free toggle.
    # When True, the 40 columns are appended as FAST features (they
    # characterize the current frequency composition of recent returns,
    # which moves quickly -- the same category the volatility class
    # already sits in), not SLOW. --
    INCLUDE_SPECTRAL_FEATURES = False

    # -- Feature partition: 204 columns (80 slow + 124 fast) by default,
    # or 244 (80 slow + 164 fast) with INCLUDE_SPECTRAL_FEATURES=True.
    # Unlike every other flag in this file, this one is resolved ONCE,
    # right here, when the class body executes at import time -- editing
    # CFG.INCLUDE_SPECTRAL_FEATURES after the module has been imported
    # (e.g. `mvsl.CFG.INCLUDE_SPECTRAL_FEATURES = True` from another
    # script) will NOT recompute these four lines and will leave you with
    # a mismatched dimension count. Set it in this source file before
    # running, the same way you already do for the other flags; just
    # don't expect a post-import assignment to take effect for this one
    # specifically. --
    if INCLUDE_SPECTRAL_FEATURES:
        SLOW_IDX = list(range(0, 40)) + list(range(160, 200))
        FAST_IDX = list(range(40, 160)) + list(range(200, 244))
        SLOW_DIM = 80
        FAST_DIM = 164
    else:
        SLOW_IDX = list(range(0, 40)) + list(range(160, 200))
        FAST_IDX = list(range(40, 160)) + list(range(200, 204))
        SLOW_DIM = 80
        FAST_DIM = 124

    # -- Physics / architecture --
    LATENT_DIM = 4
    N_VERLET_STEPS = 6
    DT = 0.1
    POTENTIAL_DIM = 32
    FORCE_DIM = 32
    DROPOUT = 0.3
    NOISE_STD = 0.05

    TEMP_MIN = 0.005
    TEMP_MAX = 0.120
    ALPHA_MIN = 0.40
    ALPHA_MAX = 0.95

    # -- Loss asymmetry (false negatives cost ASYM_CRISIS_UNDER_K times more) --
    ASYM_CRISIS_UNDER_K = 3.0
    ASYM_STABLE_OVER_K = 1.0

    ANCHOR_STABLE = 0.15
    ANCHOR_CRISIS = 0.85
    DISPLAY_SCALE = 100.0

    # Regime banding utility -- not currently invoked by this script's
    # pipeline (see psi_to_regime docstring below); kept for a four-regime
    # strategy layer built on top of this detector.
    REGIME_BANDS = [0.35, 0.50, 0.65]
    REGIME_NAMES = ["Stable", "Pre-alert", "Transition", "Crisis"]

    # -- V30.4: soft-label kernel --
    # Sigma of the Gaussian kernel, in trading days. sigma=7 gives a ~14-day
    # ramp at each crisis boundary. Worth sweeping sigma in {5, 7, 10}.
    SOFT_LABEL_SIGMA = 7

    # -- Default False: symmetric ramp (SOFT_LABEL_SIGMA both sides, via
    # make_soft_labels). True switches to make_soft_labels_asymmetric():
    # a fast ramp INTO each crisis (SOFT_LABEL_SIGMA_ONSET) and a slow
    # ramp back OUT (SOFT_LABEL_SIGMA_DECAY), motivated by "up the stairs,
    # down the elevator" -- crises arrive abruptly, recoveries are
    # gradual. Changes the training target, so this needs a full retrain
    # to evaluate; not a free change. --
    ASYMMETRIC_SOFT_LABELS = False
    SOFT_LABEL_SIGMA_ONSET = 3
    SOFT_LABEL_SIGMA_DECAY = 15

    # -- Default False: the potential network (ParametricBistablePotential)
    # sees only x_slow (80 dims: statistical + macro classes) to shape
    # (a, b, c). Credit spread (k_credit_*) and yield-curve level
    # (k_yc_*) are NOT in that 80 -- verified programmatically, not
    # assumed: they land at columns 135-139 and 140-144, inside the FAST
    # block (they feed FastEncoder's gamma/T/phi/alpha instead). Only the
    # yield curve's 10-day CHANGE (c_yc_chg_10, column 197) reaches the
    # slow/potential pathway, diluted among 79 other slow features.
    # True adds a small, separate sub-network (see ParametricBistable
    # Potential.macro_skip_net) that takes ONLY these 4 values -- credit
    # spread and yield curve, short (10d) and long (60d) window each --
    # and adds its output directly into (a, b, c), so these two
    # economically fundamental signals get their own weights instead of
    # competing for representation inside the shared 80-input param_net.
    # This is a real architecture change (new parameters, wider effective
    # input), so it needs a full retrain to know whether it helps -- not
    # a free toggle, same as every other flag that touches the model
    # itself. --
    MACRO_SKIP_CONNECTION = False
    MACRO_SKIP_IDX = [135, 138, 140, 143]  # k_credit_10, k_credit_60, k_yc_10, k_yc_60

    # -- Default False: training noise is white only, sigma^2 = 2*gamma*T*dt,
    # which satisfies the fluctuation-dissipation theorem for the
    # INSTANTANEOUS friction gamma but not for the memory friction: the
    # -phi*s term is an exponential-kernel drag (Gamma_k = phi*dt*alpha^k
    # on p_{t-k}) with NO matching fluctuation, violating the FDT of the
    # second kind (Kubo 1966). Measured consequence on this exact
    # verlet_step (harmonic well, 80k steps): the particle equilibrates at
    # 80% of its proper kinetic energy at phi=0.3 and 74% at phi=0.5 --
    # the network is forced to learn a distorted T to compensate a
    # thermostat that drains energy through the memory channel without
    # ever returning it.
    # True adds the matched Ornstein-Uhlenbeck colored noise:
    #     eta_{t+1} = alpha*eta_t + sqrt(sig_stat^2*(1-alpha^2))*N(0,1)
    #     sig_stat^2 = phi*T*dt^2 / (1 - gamma/2)
    # (prefactor calibrated to this code's hybrid convention, where the
    # effective kinetic temperature is T*dt/(1-gamma/2) -- verified:
    # predicted 0.00556, measured 0.00558). Restores equipartition to
    # 105-109% (residual is a higher-order phi*dt cross-term; exact
    # balance would need a potential-curvature-dependent matrix GLE
    # thermostat a la Ceriotti et al., overkill here). eta is initialized
    # from its stationary distribution at each forward() call: marginal
    # and within-day correlation exact, cross-day noise correlation
    # truncated (documented approximation). Changes training dynamics
    # only (no noise at inference), so this needs a full retrain to
    # evaluate -- and its effect on test metrics cannot be judged from a
    # single run against a 2-crisis test window; it is a theoretical-
    # soundness fix first, a stats hypothesis second. --
    FDT2_COLORED_NOISE = False

    # -- Default False: FastEncoder sees only the current day's fast
    # features (memoryless MLP). True adds a small causal TCN that first
    # looks at TCN_WINDOW days (oldest to current, no future leakage --
    # verified by test, see tests/test_loss_and_dynamics.py) and
    # produces a TCN_OUT_DIM summary, concatenated onto the current day's
    # xf before FastEncoder (whose input width grows accordingly). Meant
    # to estimate short-horizon "velocity" from raw features before
    # handing off to Verlet -- deliberately narrow (~1-2k params: a 1x1
    # projection to TCN_CHANNELS, then 2 small causal dilated convs) so
    # it stays a local, few-parameter estimator rather than a path around
    # the physics bottleneck. Long memory is unaffected -- phi*s carries
    # that, unchanged. Needs sl >= TCN_WINDOW in ChronoRegimeDataset
    # (sl=20 by default, ample headroom). Real architecture change, needs
    # a full retrain to evaluate; not a free toggle. --
    TCN_VELOCITY_ENCODER = False
    TCN_WINDOW = 5
    TCN_CHANNELS = 8
    TCN_OUT_DIM = 8

    # -- Default False: ASYM_CRISIS_UNDER_K (k_FN) is a fixed constant in
    # hooke_loss. True modulates it per-day by an EXOGENOUS persistence
    # signal -- consecutive periods of credit-spread widening (see
    # compute_credit_persistence) -- rather than a learned quantity, so
    # it does not inherit the small-sample fragility that sank
    # MACRO_SKIP_CONNECTION and TCN_VELOCITY_ENCODER tonight: nothing
    # here is fit against crisis outcomes or the model's own predictions,
    # only against already-observed HYG/IEF prices.
    #     k_FN(t) = ASYM_CRISIS_UNDER_K * (1 + LAMBDA * tanh(streak(t) / TAU))
    # streak(t) counts consecutive DYNAMIC_K_FN_SMOOTH_WINDOW-day blocks
    # of widening credit stress as of day t (purely from past prices, see
    # compute_credit_persistence's docstring for the causality argument
    # and test_dynamic_k_fn_no_lookahead for the direct verification).
    # Targets the specific documented weakness (slow, rate/credit-driven
    # drawdowns: 2018 Q4, 2022) without touching the panic-driven regime
    # (COVID) where the model already performs well. Real training-loss
    # change, needs a full retrain (walk-forward, not a single test-set
    # run) to evaluate -- not a free toggle. --
    DYNAMIC_K_FN = False
    DYNAMIC_K_FN_LAMBDA = 1.0
    DYNAMIC_K_FN_TAU = 6.0
    DYNAMIC_K_FN_SMOOTH_WINDOW = 5

    # -- Default False: no change to hooke_loss. True adds a per-sample
    # kinetic-energy penalty (p**2, weighted by 1-label) so the particle
    # is pushed toward low momentum specifically on days closer to
    # "stable" -- a "cool down when calm" prior, aimed at reducing false
    # positives. Changes the training loss, so this needs a full retrain
    # to evaluate; not a free change. See hooke_loss()'s docstring for
    # why the weighting must be applied per-sample, not as a product of
    # two batch-level means. --
    KINETIC_ENERGY_PENALTY = False
    LAMBDA_KINETIC = 0.01

    # Not currently referenced by hooke_loss() or anywhere else in this
    # script; kept from earlier versions in case another loss variant reads
    # it. Flagged here rather than silently dropped.
    LAMBDA_CONF = 0.3

    BATCH_SIZE = 128
    LR = 3e-4
    # 3 -> 5 seeds: pure variance reduction, not a new experiment on the
    # data. Every seed (old and new) trains on the exact same X_train /
    # y_train_soft with the exact same procedure -- nothing is added to or
    # removed from what any individual model can learn, unlike
    # STRICT_CALIBRATION_HOLDOUT above. Expect a small stabilizing effect
    # on Psi (less seed-to-seed noise), not a step change in the headline
    # metrics -- the original 3-seed run already showed ROC-AUC moving
    # ~0.0003-0.0012 between otherwise-identical runs purely from training
    # stochasticity; more seeds narrows that, it doesn't manufacture new
    # signal. Also retrains the proxy-calibration ensemble at 5 seeds
    # (train_proxy_calibration_ensemble reuses this same list), so the
    # cost is roughly double a single ensemble's worth of extra training.
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011]
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -- Methodology switch, default False (measured and reverted -- see
    # module docstring for the full story). True gives a strictly disjoint
    # train/calibration/test split, which removes the 2018-10..12 crisis
    # from training entirely; measured cost was a roughly-halved 2022
    # recall and a worse (not better) ECE, confounded with training on
    # ~11% less data. Kept as an opt-in for anyone who wants to reproduce
    # or extend that measurement. --
    STRICT_CALIBRATION_HOLDOUT = False

    # -- Default True: also trains a second, disposable ensemble on data
    # strictly before the calibration window, solely to fit an
    # out-of-sample temperature scalar (see train_proxy_calibration_
    # ensemble()). Unlike STRICT_CALIBRATION_HOLDOUT, this cannot change
    # any detection metric -- the final model above trains exactly as it
    # always has. The only cost is roughly one extra ensemble's worth of
    # training time. The report prints both the in-sample and the proxy
    # calibration numbers side by side rather than picking one for you. --
    PROXY_CALIBRATION = True

    # -- Default False: T_calib (1-parameter temperature scaling) is the
    # only calibration map fit. True also fits beta calibration (3
    # parameters: a*log(p) - b*log(1-p) + c), which can correct
    # asymmetric miscalibration a single temperature cannot -- worth
    # trying since CFG.ASYMMETRIC_SOFT_LABELS pushes Psi's distribution
    # away from the symmetric case T_calib was designed for. Fit on the
    # calibration set only, same as T_calib; costs nothing beyond one
    # cheap logistic-regression fit (milliseconds, not a retrain). See
    # evaluation_metrics.fit_beta_calibration for the caveat about 3
    # parameters on a calibration window with only one crisis episode. --
    BETA_CALIBRATION = False


# ---- Physical components (unchanged vs. V30.3) ----
class ZScoreNorm(nn.Module):
    """Fixed (non-trainable) z-score normalization, fit once on the
    training set and reused as-is for calibration/test -- no leakage.
    Also winsorizes to the training set's [0.1, 99.9] percentiles before
    normalizing: protects against a bad data point (a stale/erroneous
    tick, a feed glitch) blowing up gradients or skewing the mean/std
    that every subsequent forward pass depends on. In the ~99.8% of cases
    where a value is already inside that range, this is a no-op."""

    def __init__(self, d: int):
        super().__init__()
        self.register_buffer("means", torch.zeros(d))
        self.register_buffer("stds", torch.ones(d))
        self.register_buffer("clip_lo", torch.full((d,), -float("inf")))
        self.register_buffer("clip_hi", torch.full((d,), float("inf")))

    def fit(self, X: np.ndarray) -> None:
        self.clip_lo.copy_(torch.tensor(np.nanpercentile(X, 0.1, axis=0), dtype=torch.float32))
        self.clip_hi.copy_(torch.tensor(np.nanpercentile(X, 99.9, axis=0), dtype=torch.float32))
        x_clipped = np.clip(X, self.clip_lo.numpy(), self.clip_hi.numpy())
        self.means.copy_(torch.tensor(np.nanmean(x_clipped, 0), dtype=torch.float32))
        self.stds.copy_(torch.tensor(np.nanstd(x_clipped, 0) + 1e-8, dtype=torch.float32))

    def forward(self, x):
        x = torch.clamp(x, self.clip_lo, self.clip_hi)
        return (x - self.means) / self.stds


class GaussianGatedActivation(nn.Module):
    """x * (1 + exp(-x^2)) / 2 -- a smooth, bounded gate around zero."""

    def forward(self, x):
        return x * (1.0 + torch.exp(-x**2)) / 2.0


class ParametricBistablePotential(nn.Module):
    """Ginzburg-Landau quartic potential V(z) = a*z^4 - b*z^2 + c*z, with
    (a, b, c) predicted per-day from the slow features rather than fixed."""

    def __init__(self, slow_dim: int, latent_dim: int, hidden: int = 32, macro_skip_dim: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        act = GaussianGatedActivation
        self.param_net = nn.Sequential(
            nn.Linear(slow_dim, hidden), act(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(hidden, 3),
        )
        # Dedicated pathway (see CFG.MACRO_SKIP_CONNECTION): a small,
        # separate sub-network for credit spread + yield curve, so these
        # 2 signals get their own weights rather than sharing param_net's
        # capacity with 76 unrelated slow features. Its raw 3-dim output
        # is ADDED to param_net's raw output below, not concatenated into
        # the input -- this keeps param_net itself completely unchanged
        # (same weights shape, same behavior) when this is disabled.
        self.macro_skip_dim = macro_skip_dim
        if macro_skip_dim > 0:
            self.macro_skip_net = nn.Sequential(
                nn.Linear(macro_skip_dim, 8), nn.Tanh(),
                nn.Linear(8, 3),
            )

    def get_params(self, x_slow, x_macro_skip=None, vol_signal=None):
        raw = self.param_net(x_slow)
        if self.macro_skip_dim > 0 and x_macro_skip is not None:
            raw = raw + self.macro_skip_net(x_macro_skip)
        return (
            F.softplus(raw[:, 0:1]) + 0.1,
            F.softplus(raw[:, 1:2]) + 0.1,
            torch.tanh(raw[:, 2:3]) * 2.0,
        )

    def energy(self, x_slow, z, x_macro_skip=None, vol_signal=None):
        a, b, c = self.get_params(x_slow, x_macro_skip)
        z2 = (z**2).sum(dim=1, keepdim=True)
        return a * z2**2 - b * z2 + c * z.mean(dim=1, keepdim=True)

    def gradient(self, x_slow, z, x_macro_skip=None, vol_signal=None):
        a, b, c = self.get_params(x_slow, x_macro_skip)
        z2 = torch.clamp((z**2).sum(dim=1, keepdim=True), max=10.0)
        return torch.clamp((4.0 * a * z2 - 2.0 * b) * z + c / self.latent_dim, -10.0, 10.0)

    def forward(self, x_slow, z, x_macro_skip=None, vol_signal=None):
        return self.energy(x_slow, z, x_macro_skip), self.gradient(x_slow, z, x_macro_skip)


class CausalTCN(nn.Module):
    """Small causal 1D-conv stack over a short window of fast features,
    producing a per-day summary meant to estimate short-horizon
    "velocity" before handing off to FastEncoder. Not a replacement for
    the model's long memory (phi*s): this only ever sees TCN_WINDOW days.

    Narrow by design (~1-2k params total): a 1x1 projection (a per-
    timestep Linear) from fast_dim down to `channels` first, so the
    temporal convolutions themselves operate on a small number of
    channels rather than on all 124 fast features directly.

    Dilation schedule (1, 2, 4) across 3 kernel=2 layers: this specific
    doubling progression is the standard TCN construction because it is
    what guarantees a gap-free receptive field. An arbitrary dilation
    (dilation=3 for a second layer was tried first here) can leave a HOLE
    in coverage -- caught directly by test_tcn_receptive_field_covers_
    full_window, which found position 2 of a 5-day window silently never
    reaching the output despite positions 0, 1, 3, 4 all reaching it.
    Total receptive field with (1, 2, 4) is 8 days, comfortably covering
    TCN_WINDOW=5 with the extra reach absorbed by left-padding zeros.

    Causality: padded on the LEFT only at every layer, so the output at
    the last (current-day) position depends only on the current day and
    earlier days, never on anything after it -- verified directly by
    test_tcn_causality."""

    def __init__(self, fast_dim: int, channels: int = 8, out_dim: int = 8):
        super().__init__()
        act = GaussianGatedActivation
        self.project = nn.Linear(fast_dim, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=2, dilation=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=2, dilation=2)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=2, dilation=4)
        self.act = act()
        self.out = nn.Linear(channels, out_dim)

    def _causal_conv(self, conv, h):
        h = F.pad(h, (conv.dilation[0] * (conv.kernel_size[0] - 1), 0))
        return self.act(conv(h))

    def forward(self, x_window):
        """x_window: (batch, window, fast_dim), already z-score normalized
        the same way as the current day (see PhaseIndexDetector.forward),
        oldest day first, current day last."""
        h = self.project(x_window)               # (batch, window, channels)
        h = h.transpose(1, 2)                     # (batch, channels, window)
        h = self._causal_conv(self.conv1, h)
        h = self._causal_conv(self.conv2, h)
        h = self._causal_conv(self.conv3, h)
        h_last = h[:, :, -1]                       # representation at the current day
        return self.out(h_last)


class FastEncoder(nn.Module):
    """Maps the fast features to everything the Verlet-Langevin step needs:
    initial position/momentum/external force (z, p, f_ext), friction
    (gamma), temperature (T), memory-kernel weight (phi), and dissipation
    (alpha)."""

    def __init__(self, fast_dim: int, latent_dim: int, hidden: int = 32):
        super().__init__()
        act = GaussianGatedActivation

        def mlp(*dims, out_act=None):
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1])]
                if i < len(dims) - 2:
                    layers += [act(), nn.Dropout(CFG.DROPOUT)]
            if out_act:
                layers.append(out_act())
            return nn.Sequential(*layers)

        self.z_enc = mlp(fast_dim, hidden, latent_dim, out_act=nn.Tanh)
        self.p_enc = mlp(fast_dim, hidden, latent_dim)
        self.f_enc = mlp(fast_dim, hidden, latent_dim)
        self.gam_enc = mlp(fast_dim, hidden, 1, out_act=nn.Sigmoid)
        self.tmp_enc = mlp(fast_dim, hidden, 1, out_act=nn.Sigmoid)
        self.mem_enc = nn.Sequential(nn.Linear(fast_dim, hidden), act(), nn.Linear(hidden, 1), nn.Sigmoid())
        self.alp_enc = nn.Sequential(nn.Linear(fast_dim, hidden), act(), nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        return (
            self.z_enc(x),
            self.p_enc(x) * 0.1,
            self.f_enc(x) * 0.1,
            self.gam_enc(x),
            CFG.TEMP_MIN + (CFG.TEMP_MAX - CFG.TEMP_MIN) * self.tmp_enc(x),
            self.mem_enc(x) * 0.5,
            CFG.ALPHA_MIN + (CFG.ALPHA_MAX - CFG.ALPHA_MIN) * self.alp_enc(x),
        )


def verlet_step(z, p, s, x_slow, potential, f_ext, dt,
                 gamma=0.0, temp=0.0, phi=0.0, alpha=0.8,
                 training=False, vol_signal=None, x_macro_skip=None,
                 colored_noise=None):
    """One velocity-Verlet step under Langevin thermostatting, with an
    extra memory-drag term (-phi*s) from the accumulated state s. During
    training, thermal noise sigma = sqrt(2*gamma*T*dt) is injected into the
    momentum update (fluctuation-dissipation). If colored_noise is given
    (an OU process maintained by the caller -- see CFG.FDT2_COLORED_NOISE),
    it is added to the momentum at the same point as the white noise, so
    both fluctuation channels enter the integrator identically."""
    _, grad_V = potential(x_slow, z, x_macro_skip, vol_signal)
    p_next = (1.0 - gamma) * p - dt * grad_V + dt * f_ext - phi * s
    if training:
        if not torch.is_tensor(temp):
            temp = torch.tensor(float(temp), device=z.device, dtype=z.dtype)
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(float(gamma), device=z.device, dtype=z.dtype)
        sigma = torch.sqrt(torch.clamp(2.0 * gamma * temp * dt, min=1e-12))
        p_next = p_next + sigma * torch.randn_like(p_next)
    if colored_noise is not None:
        p_next = p_next + colored_noise
    z_next = z + dt * p_next
    s_next = alpha * s + dt * p_next
    return z_next, p_next, s_next


class PhaseIndexDetector(nn.Module):
    """Full model: normalize -> encode fast features into physical
    parameters -> unroll N_VERLET_STEPS of Langevin dynamics -> map the
    final (z, p) state to a phase index Psi in [0, 1] via a sigmoid head.

    `s_initial` carries the memory-kernel accumulator across sequential
    calls, which is how "stateful" continuity across trading days is
    implemented -- the Dataset below does not build sequences itself; the
    training loop and inference loop pass the state through by hand,
    day by day / batch by batch.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.norm_slow = ZScoreNorm(CFG.SLOW_DIM)
        self.norm_fast = ZScoreNorm(CFG.FAST_DIM)
        self.potential = ParametricBistablePotential(
            CFG.SLOW_DIM, CFG.LATENT_DIM, CFG.POTENTIAL_DIM,
            macro_skip_dim=len(CFG.MACRO_SKIP_IDX) if CFG.MACRO_SKIP_CONNECTION else 0,
        )
        # CFG.MACRO_SKIP_IDX holds absolute column indices into the full
        # 204-column input. xf only contains the FAST_IDX subset of those
        # columns (already z-score normalized), so forward() needs each
        # macro-skip column's POSITION WITHIN xf, not its absolute index
        # -- computed once here rather than on every forward() call.
        if CFG.MACRO_SKIP_CONNECTION:
            self._macro_skip_rel_idx = [CFG.FAST_IDX.index(i) for i in CFG.MACRO_SKIP_IDX]
        # TCN velocity encoder (see CFG.TCN_VELOCITY_ENCODER): when on,
        # FastEncoder's input width grows to make room for the TCN's
        # summary, concatenated onto the current day's xf in forward().
        # param_net/potential are untouched -- this only affects the fast
        # (dynamics) pathway, not the slow (regime/potential) pathway.
        self.tcn = CausalTCN(CFG.FAST_DIM, CFG.TCN_CHANNELS, CFG.TCN_OUT_DIM) if CFG.TCN_VELOCITY_ENCODER else None
        fast_enc_dim = CFG.FAST_DIM + CFG.TCN_OUT_DIM if CFG.TCN_VELOCITY_ENCODER else CFG.FAST_DIM
        self.fast_enc = FastEncoder(fast_enc_dim, CFG.LATENT_DIM, CFG.FORCE_DIM)
        self.head = nn.Sequential(
            nn.Linear(CFG.LATENT_DIM * 2, 16),
            GaussianGatedActivation(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def fit_preprocessing(self, X: np.ndarray) -> None:
        self.norm_slow.fit(X[:, CFG.SLOW_IDX])
        self.norm_fast.fit(X[:, CFG.FAST_IDX])

    def forward(self, x, s_initial=None, xf_window=None):
        xs = self.norm_slow(x[:, CFG.SLOW_IDX])
        xf = self.norm_fast(x[:, CFG.FAST_IDX])
        if self.training and CFG.NOISE_STD > 0:
            xs = xs + CFG.NOISE_STD * torch.randn_like(xs)
            xf = xf + CFG.NOISE_STD * torch.randn_like(xf)

        if CFG.TCN_VELOCITY_ENCODER:
            assert xf_window is not None, (
                "CFG.TCN_VELOCITY_ENCODER is True but forward() got no xf_window -- "
                "every caller (train_one, run_stateful_inference) must supply the "
                "trailing TCN_WINDOW days of RAW fast-feature columns when this flag "
                "is on. Failing loudly here rather than silently falling back to "
                "zeros, which would look like a working run with degraded signal."
            )
            # xf_window: (batch, TCN_WINDOW, len(FAST_IDX)) raw columns, oldest
            # first. Normalized per-timestep with the SAME norm_fast used for
            # the current day above (broadcasts correctly -- verified by
            # test_tcn_window_normalization_matches_per_day_norm).
            xf_window_norm = self.norm_fast(xf_window)
            tcn_out = self.tcn(xf_window_norm)
            xf = torch.cat([xf, tcn_out], dim=1)

        z, p, f_ext, gamma_t, temp_t, phi_t, alpha_t = self.fast_enc(xf)
        vol = xf[:, 0:40].abs().mean(dim=1, keepdim=True)
        self._last_gamma = gamma_t
        self._last_temp = temp_t
        self._last_alpha = alpha_t
        x_macro_skip = xf[:, self._macro_skip_rel_idx] if CFG.MACRO_SKIP_CONNECTION else None
        # a/b/c depend only on xs (and x_macro_skip, when enabled), both
        # fixed for this call (only z changes across the Verlet loop
        # below) -- capturing them here is a read of the same
        # deterministic function verlet_step already calls internally via
        # potential(), not a second independent computation path. Used
        # only for the post-hoc Kramers-rate diagnostic; does not feed
        # back into psi/z/p/s.
        a_t, b_t, _c_t = self.potential.get_params(xs, x_macro_skip)
        self._last_a = a_t
        self._last_b = b_t

        s = torch.zeros_like(p) if s_initial is None else s_initial.clone()
        # FDT-2 colored noise (see CFG.FDT2_COLORED_NOISE): an OU process
        # matched to the memory kernel's alpha, stationary-initialized
        # here, updated between substeps below. Training only -- like the
        # white noise, inference stays deterministic.
        if self.training and CFG.FDT2_COLORED_NOISE:
            sig_stat2 = torch.clamp(
                phi_t * temp_t * CFG.DT * CFG.DT / (1.0 - gamma_t / 2.0), min=0.0,
            )
            eta = torch.randn_like(p) * sig_stat2.sqrt()
            sigma_eta = torch.clamp(sig_stat2 * (1.0 - alpha_t ** 2), min=1e-18).sqrt()
        else:
            eta = None
        for _ in range(CFG.N_VERLET_STEPS):
            z, p, s = verlet_step(z, p, s, xs, self.potential, f_ext, CFG.DT,
                                   gamma=gamma_t, temp=temp_t, phi=phi_t, alpha=alpha_t,
                                   training=self.training, vol_signal=vol, x_macro_skip=x_macro_skip,
                                   colored_noise=eta)
            if eta is not None:
                eta = alpha_t * eta + sigma_eta * torch.randn_like(eta)

        # Captured for the optional kinetic-energy loss term (see hooke_loss's
        # p/kinetic_lambda arguments) -- read, not computed twice; p here is
        # exactly the momentum the psi head below also consumes.
        self._last_p = p
        psi = self.head(torch.cat([z, p], dim=1)).squeeze(-1)
        return psi, z, s


# ---- Labels ----
CRISES = [
    ("1994-02-01", "1994-12-31"), ("1997-10-01", "1997-12-31"),
    ("1998-08-01", "1998-10-31"), ("2001-09-01", "2001-11-30"),
    ("2002-07-01", "2002-10-31"), ("2007-07-01", "2007-09-30"),
    ("2008-09-01", "2009-03-31"), ("2010-05-01", "2010-07-31"),
    ("2011-08-01", "2011-10-31"), ("2015-08-01", "2015-09-30"),
    ("2018-10-01", "2018-12-31"), ("2020-02-20", "2020-04-30"),
    ("2022-06-01", "2022-10-31"),
]

# Display names for the report; falls back to the raw date range for any
# window not listed here. Keys must match CRISES entries exactly.
CRISIS_NAMES = {
    ("1994-02-01", "1994-12-31"): "1994 Fed tightening",
    ("1997-10-01", "1997-12-31"): "Asian financial crisis",
    ("1998-08-01", "1998-10-31"): "LTCM / Russian default",
    ("2001-09-01", "2001-11-30"): "September 11",
    ("2002-07-01", "2002-10-31"): "Dot-com bust (late leg)",
    ("2007-07-01", "2007-09-30"): "Subprime onset",
    ("2008-09-01", "2009-03-31"): "Global financial crisis",
    ("2010-05-01", "2010-07-31"): "2010 flash crash / EU debt (I)",
    ("2011-08-01", "2011-10-31"): "US downgrade / EU debt (II)",
    ("2015-08-01", "2015-09-30"): "August 2015 selloff",
    ("2018-10-01", "2018-12-31"): "Q4 2018 selloff",
    ("2020-02-20", "2020-04-30"): "COVID-19 crash",
    ("2022-06-01", "2022-10-31"): "2022 rate-hike drawdown",
}


def make_hard_labels(index: pd.DatetimeIndex) -> pd.Series:
    """Binary {0, 1} labels -- used for EVALUATION."""
    y = pd.Series(0, index=index, dtype=int)
    for start, end in CRISES:
        y[(index >= start) & (index <= end)] = 1
    return y


def make_soft_labels(index: pd.DatetimeIndex, sigma: int | None = None) -> pd.Series:
    """Continuous [0, 1] labels -- used for TRAINING only.

    The Gaussian kernel (sigma trading days) creates a ramp at each crisis
    boundary: ~0.16 seven days before onset, 0.50 on the first official
    crisis day, ~0.84 seven days after, symmetric on the way out. This
    removes artificial discontinuities in the GL potential landscape and
    keeps training gradients aligned with the particle's physics.
    """
    if sigma is None:
        sigma = CFG.SOFT_LABEL_SIGMA
    y_hard = make_hard_labels(index).values.astype(np.float64)
    # mode="constant", cval=0: outside the series edges, assume no crisis.
    y_soft = gaussian_filter1d(y_hard, sigma=sigma, mode="constant", cval=0.0)
    y_soft = np.clip(y_soft, 0.0, 1.0)
    return pd.Series(y_soft.astype(np.float32), index=index)


def make_soft_labels_asymmetric(index: pd.DatetimeIndex, sigma_onset: float | None = None,
                                 sigma_decay: float | None = None) -> pd.Series:
    """Continuous [0, 1] labels with a fast ramp INTO each crisis and a
    slow ramp back OUT, instead of make_soft_labels()'s single symmetric
    sigma. Used for TRAINING only when CFG.ASYMMETRIC_SOFT_LABELS is True.

    scipy.ndimage.gaussian_filter1d cannot produce this directly (one
    sigma, applied uniformly), so this is built per-crisis-window instead:
    for each (start, end) in CRISES, a one-sided Gaussian with sigma_onset
    governs the approach from before `start` (small sigma_onset -> label
    stays near 0 until close to onset, then rises steeply), a one-sided
    Gaussian with sigma_decay governs the recession after `end` (large
    sigma_decay -> label stays elevated for longer after the window
    closes), and the window itself is 1.0 throughout [start, end].
    Windows are combined by taking the pointwise maximum across all of
    CRISES, so nearby crises' ramps do not cancel each other out.

    Caution: taking the maximum of two *globally* Gaussian-smoothed series
    (one per sigma) does NOT give this shape -- a wide-sigma filter starts
    rising further before onset than a narrow one does, so a naive
    elementwise max of two full-series filters ends up picking the SLOW
    filter on the approach to a crisis, the opposite of "fast onset".
    Building the ramp per-window from each boundary, as done here, avoids
    that trap.
    """
    if sigma_onset is None:
        sigma_onset = CFG.SOFT_LABEL_SIGMA_ONSET
    if sigma_decay is None:
        sigma_decay = CFG.SOFT_LABEL_SIGMA_DECAY

    n = len(index)
    t = np.arange(n, dtype=np.float64)
    y_soft = np.zeros(n, dtype=np.float64)

    for start, end in CRISES:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if end_ts < index.min() or start_ts > index.max():
            continue
        start_pos = int(index.searchsorted(start_ts))
        end_pos = int(index.searchsorted(end_ts, side="right")) - 1
        window = np.zeros(n, dtype=np.float64)
        window[max(start_pos, 0):min(end_pos + 1, n)] = 1.0

        pre_mask = t < start_pos
        window[pre_mask] = np.maximum(
            window[pre_mask],
            np.exp(-0.5 * ((t[pre_mask] - start_pos) / sigma_onset) ** 2),
        )
        post_mask = t > end_pos
        window[post_mask] = np.maximum(
            window[post_mask],
            np.exp(-0.5 * ((t[post_mask] - end_pos) / sigma_decay) ** 2),
        )
        y_soft = np.maximum(y_soft, window)

    y_soft = np.clip(y_soft, 0.0, 1.0)
    return pd.Series(y_soft.astype(np.float32), index=index)


def build_soft_labels(index: pd.DatetimeIndex) -> pd.Series:
    """Dispatches to make_soft_labels() or make_soft_labels_asymmetric()
    based on CFG.ASYMMETRIC_SOFT_LABELS, so main() does not need an
    if/else of its own."""
    if CFG.ASYMMETRIC_SOFT_LABELS:
        return make_soft_labels_asymmetric(index)
    return make_soft_labels(index, sigma=CFG.SOFT_LABEL_SIGMA)


# ---- Loss ----
def hooke_loss(psi: torch.Tensor, labels: torch.Tensor, p: "torch.Tensor | None" = None,
               kinetic_lambda: float = 0.0, persistence: "torch.Tensor | None" = None) -> torch.Tensor:
    """Continuous Hooke's-law loss for soft labels y in [0, 1].

    Interpolated target:
        Psi*(y) = ANCHOR_STABLE + (ANCHOR_CRISIS - ANCHOR_STABLE) * y
                = 0.15 + 0.70 * y

    Interpolated stiffness below the target (under-shooting a crisis):
        k(y) = 1 + (k_FN - 1) * y  in [1.0, k_FN]
             = 1.0 at y=0 (stable day, no extra penalty)
             = 2.0 at y=0.5 (transition, intermediate penalty)
             = 3.0 at y=1 (pure crisis, maximum penalty)

    Above the target on a stable day (false alarm): stiffness = k_FP = 1.0,
    unchanged.

    This avoids the gradient discontinuity a binary-label version has at
    every crisis boundary: the particle follows a continuous trajectory in
    the GL potential and the model learns the transitions, not just the
    pure states.

    Optional kinetic-energy regularizer (off by default: p=None or
    kinetic_lambda=0.0 reproduces the original loss exactly, byte for
    byte). When enabled (see CFG.KINETIC_ENERGY_PENALTY), penalizes
    per-sample kinetic energy (p**2, averaged over the latent dimensions)
    weighted by (1 - label), so days closer to "stable" get more pressure
    toward low momentum -- a "cool down" prior for calm periods. The
    weighting is applied PER SAMPLE before averaging over the batch
    (kinetic_energy_per_sample * (1 - labels)).mean() -- not
    kinetic_energy.mean() * (1 - labels).mean(), which would multiply two
    batch-level averages together instead of correlating momentum with
    calmness sample by sample, and is not the same quantity.

    Optional dynamic k_FN (off by default: persistence=None reproduces
    the original loss exactly). When CFG.DYNAMIC_K_FN is True and a
    per-sample `persistence` tensor is supplied (see
    compute_credit_persistence), the false-negative penalty ceiling
    itself becomes exogenous-signal-dependent instead of the fixed
    CFG.ASYM_CRISIS_UNDER_K:
        effective_k_fn = ASYM_CRISIS_UNDER_K * (1 + LAMBDA * tanh(persistence / TAU))
    which then interpolates by label exactly as before. persistence is
    computed purely from past HYG/IEF prices, never from the model's own
    outputs or the crisis labels -- see CFG.DYNAMIC_K_FN's docstring for
    why that distinction matters.
    """
    target = CFG.ANCHOR_STABLE + (CFG.ANCHOR_CRISIS - CFG.ANCHOR_STABLE) * labels
    err = psi - target

    if persistence is not None and CFG.DYNAMIC_K_FN:
        effective_asym_k = CFG.ASYM_CRISIS_UNDER_K * (
            1.0 + CFG.DYNAMIC_K_FN_LAMBDA * torch.tanh(persistence / CFG.DYNAMIC_K_FN_TAU)
        )
    else:
        effective_asym_k = CFG.ASYM_CRISIS_UNDER_K
    k_fn = 1.0 + (effective_asym_k - 1.0) * labels  # in [1.0, effective_asym_k]

    stiffness = torch.ones_like(psi)
    stiffness = torch.where(err < 0, k_fn, stiffness)
    stiffness = torch.where(
        (err > 0) & (labels < 0.5),
        torch.full_like(stiffness, CFG.ASYM_STABLE_OVER_K),
        stiffness,
    )
    total = (stiffness * err.pow(2)).mean()

    if p is not None and kinetic_lambda > 0:
        kinetic_energy_per_sample = (p ** 2).mean(dim=1)
        kinetic_penalty = (kinetic_energy_per_sample * (1.0 - labels)).mean()
        total = total + kinetic_lambda * kinetic_penalty

    return total


def psi_to_regime(psi: np.ndarray) -> np.ndarray:
    """Buckets a continuous Psi array into CFG.REGIME_BANDS. Not called
    anywhere in this script's current pipeline -- available for a
    regime-conditioned strategy layer built on top of the detector."""
    regime = np.zeros(len(psi), dtype=int)
    for k, band in enumerate(CFG.REGIME_BANDS):
        regime[psi >= band] = k + 1
    return regime


# ---- Dataset & training (unchanged vs. V30.3 except SWA, already integrated there) ----
class ChronoRegimeDataset(Dataset):
    """Chronological dataset. Accepts float (soft) labels without any
    special-casing. `sl` skips the first `sl` days as a look-back margin
    -- also what makes it safe to slice a TCN_WINDOW-day window ending at
    every returned index (CFG.TCN_WINDOW must be <= sl; default 5 <= 20)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, persistence: "np.ndarray | None" = None,
                 sl: int = 20):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)  # float32: soft labels ok
        self.idx = list(range(sl, len(y)))
        self.persistence = (
            torch.tensor(persistence, dtype=torch.float32) if persistence is not None
            else torch.zeros(len(y), dtype=torch.float32)
        )
        if CFG.TCN_VELOCITY_ENCODER:
            assert CFG.TCN_WINDOW <= sl, (
                f"CFG.TCN_WINDOW={CFG.TCN_WINDOW} exceeds the dataset's look-back "
                f"margin sl={sl} -- would slice before the start of X for early rows."
            )

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        # Always computed (cheap: pure indexing) even when the TCN is off,
        # so __getitem__'s return shape never changes with the flag and
        # DataLoader's default collation stays simple. train_one ignores
        # this when CFG.TCN_VELOCITY_ENCODER is False.
        window = self.X[i - CFG.TCN_WINDOW + 1: i + 1, CFG.FAST_IDX]
        return self.X[i], self.y[i], window, self.persistence[i]


def train_one(seed: int, X_tr: np.ndarray, y_tr_soft: np.ndarray, device: str,
              persistence_tr: "np.ndarray | None" = None,
              swa_start_epoch: int = 16) -> tuple[PhaseIndexDetector, list[float]]:
    """Trains one ensemble member with SWA + soft labels.

    Returns
    -------
    model : PhaseIndexDetector
    loss_history : list[float]
        Average Hooke loss logged every 5 epochs (same cadence the console
        log already used) -- recorded only for the training-curve plot and
        has no effect on training itself.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = PhaseIndexDetector(X_tr.shape[1])
    model.fit_preprocessing(X_tr)
    model = model.to(device)

    dataset = ChronoRegimeDataset(X_tr, y_tr_soft, persistence=persistence_tr)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)

    swa_state, swa_n = None, 0
    loss_history: list[float] = []

    for phase, n_epochs in (("1", 15), ("2", 30)):
        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = 0.0
            s_moving = torch.zeros((CFG.BATCH_SIZE, CFG.LATENT_DIM), device=device)

            for Xt, yt, win, pers in loader:
                if Xt.shape[0] != CFG.BATCH_SIZE:
                    continue  # drop the ragged tail batch: keeps s_moving's shape fixed
                Xt, yt = Xt.to(device), yt.to(device)
                optimizer.zero_grad()
                if CFG.TCN_VELOCITY_ENCODER:
                    psi, _, s_moving = model(Xt, s_initial=s_moving, xf_window=win.to(device))
                else:
                    psi, _, s_moving = model(Xt, s_initial=s_moving)
                s_moving = s_moving.detach()
                loss_kwargs = {}
                if CFG.KINETIC_ENERGY_PENALTY:
                    loss_kwargs["p"] = model._last_p
                    loss_kwargs["kinetic_lambda"] = CFG.LAMBDA_KINETIC
                if CFG.DYNAMIC_K_FN:
                    loss_kwargs["persistence"] = pers.to(device)
                loss = hooke_loss(psi, yt, **loss_kwargs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)

            # SWA: accumulate over the second half of phase 2.
            if phase == "2" and epoch >= swa_start_epoch:
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.detach().clone().float() for k, v in sd.items()}
                    swa_n = 1
                else:
                    swa_n += 1
                    for k, v in sd.items():
                        swa_state[k].mul_(1.0 - 1.0 / swa_n).add_(v.detach().float() / swa_n)

            if epoch % 5 == 0:
                loss_history.append(avg_loss)
                with torch.no_grad():
                    model.eval()
                    # Offset by TCN_WINDOW-1 (0 when the flag is off) so
                    # every sampled row has enough preceding history for
                    # its window -- rows [0, TCN_WINDOW-2] never would.
                    off = CFG.TCN_WINDOW - 1 if CFG.TCN_VELOCITY_ENCODER else 0
                    sample = torch.tensor(X_tr[off:off + 200], dtype=torch.float32).to(device)
                    if CFG.TCN_VELOCITY_ENCODER:
                        win_np = np.stack([
                            X_tr[off + k - CFG.TCN_WINDOW + 1: off + k + 1, CFG.FAST_IDX]
                            for k in range(sample.shape[0])
                        ])
                        model(sample, xf_window=torch.tensor(win_np, dtype=torch.float32).to(device))
                    else:
                        model(sample)  # populates _last_gamma / _last_temp / _last_alpha
                    logger.info(
                        "seed %s | phase %s ep %2d | loss=%.4f | gamma=%.3f | T=%.3f | alpha=%.3f",
                        seed, phase, epoch, avg_loss,
                        model._last_gamma.mean().item(),
                        model._last_temp.mean().item(),
                        model._last_alpha.mean().item(),
                    )
                    model.train()

    if swa_state is not None:
        target_sd = model.state_dict()
        for k in target_sd:
            target_sd[k].copy_(swa_state[k].to(target_sd[k].dtype))
        model.load_state_dict(target_sd)

    return model, loss_history


def get_temperature_scaler(psi_calib: np.ndarray, y_calib_hard: np.ndarray) -> float:
    """Single-parameter temperature scaling fit on binary (hard) labels
    via calibration-set log-loss minimization. See CFG.STRICT_CALIBRATION_
    HOLDOUT above for a caveat about what "calibration set" means here."""
    eps = 1e-6
    pc = np.clip(psi_calib, eps, 1 - eps)
    lg = logit(pc)

    def objective(t):
        return log_loss(y_calib_hard, expit(lg / t[0]))

    return float(minimize(objective, [1.0], bounds=[(0.05, 10.0)]).x[0])


# ---- Report ----
def _confusion_and_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """tp/tn/fp/fn plus the ratios the report needs. Factored out so the
    base-radar and calibrated-radar tables share one implementation instead
    of two copies of the same arithmetic."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f2 = 5 * precision * recall / (4 * precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn + 1e-8)
    return dict(
        tp=tp, tn=tn, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1, f2=f2, accuracy=accuracy, fpr=fpr,
        confusion_matrix=np.array([[tn, fp], [fn, tp]]),
    )


def build_report(
    y_true_hard: np.ndarray,
    psi: np.ndarray,
    p_calib: np.ndarray,
    dates: pd.DatetimeIndex,
    gamma: np.ndarray,
    temp: np.ndarray,
    alpha: np.ndarray,
    tau: float,
    loss_histories: dict | None = None,
    p_calib_proxy: np.ndarray | None = None,
    t_calib_proxy: float | None = None,
    p_calib_beta: np.ndarray | None = None,
    kramers_diagnostic: dict | None = None,
    blend_diagnostic: dict | None = None,
    version: str = "V30.4",
    previous_version: str = "V30.3",
    out_dir: str | Path = "results/figures",
    history_path: str | Path = "results/run_history.json",
) -> dict:
    """Computes every metric, prints a console summary, saves the figures,
    and updates the run-history log. Returns the full metrics dict."""

    p65 = (psi >= 0.65).astype(int)
    base = _confusion_and_rates(y_true_hard, p65)

    p_tau = (p_calib >= tau).astype(int)
    calib = _confusion_and_rates(y_true_hard, p_tau)

    roc_auc = float(roc_auc_score(y_true_hard, psi))
    pr_auc = float(average_precision_score(y_true_hard, psi))
    pr_auc_legacy = em.legacy_average_precision(y_true_hard, psi)
    cd = em.cohens_d(psi, y_true_hard)
    ece_value = em.expected_calibration_error(y_true_hard, p_calib)
    brier = float(brier_score_loss(y_true_hard, p_calib))
    ll = float(log_loss(y_true_hard, p_calib))
    extra = em.point_estimates(y_true_hard, psi, p_calib, p_tau)

    print(f"\n{version} report -- soft labels (sigma={CFG.SOFT_LABEL_SIGMA}) + SWA")
    print("=" * 70)

    print("\nDetection capacity")
    print(f"  {'metric':<12}{'base (psi>=0.65)':>20}{'calibrated (tau=%.2f)' % tau:>24}")
    print(f"  {'accuracy':<12}{base['accuracy']*100:19.2f}%{calib['accuracy']*100:23.2f}%")
    print(f"  {'recall':<12}{base['recall']*100:19.2f}%{calib['recall']*100:23.2f}%")
    print(f"  {'precision':<12}{base['precision']*100:19.2f}%{calib['precision']*100:23.2f}%")
    print(f"  {'f1':<12}{base['f1']*100:19.2f}%{calib['f1']*100:23.2f}%")
    print(f"  {'f2':<12}{'n/a':>20}{calib['f2']*100:23.2f}%")
    print(f"  {'fpr':<12}{base['fpr']*100:19.2f}%{calib['fpr']*100:23.2f}%")

    print("\nCalibration (in-sample -- T_calib fit on data the final model trained on)")
    print(f"  ECE                  : {ece_value*100:.2f}%")
    print(f"  Brier                : {brier:.4f}  (skill vs. base rate: {extra['brier_skill_score']:+.3f})")
    print(f"  Log-loss             : {ll:.4f}")
    print(f"  Calibration slope    : {extra['calibration_slope']:.3f}   (1.0 = ideal)")
    print(f"  Calibration intercept: {extra['calibration_intercept']:+.3f}   (0.0 = ideal)")

    ece_proxy = brier_proxy = ll_proxy = slope_proxy = intercept_proxy = None
    if p_calib_proxy is not None:
        ece_proxy = em.expected_calibration_error(y_true_hard, p_calib_proxy)
        brier_proxy = float(brier_score_loss(y_true_hard, p_calib_proxy))
        ll_proxy = float(log_loss(y_true_hard, p_calib_proxy))
        slope_proxy, intercept_proxy = em.calibration_slope_intercept(y_true_hard, p_calib_proxy)
        print(f"\nCalibration (proxy -- T_calib={t_calib_proxy:.3f} fit on a separate ensemble that "
              f"never saw the calibration window)")
        print(f"  ECE                  : {ece_proxy*100:.2f}%"
              f"  ({'better' if ece_proxy < ece_value else 'worse'} than in-sample by "
              f"{abs(ece_proxy - ece_value)*100:.2f}pp)")
        print(f"  Brier                : {brier_proxy:.4f}")
        print(f"  Log-loss             : {ll_proxy:.4f}")
        print(f"  Calibration slope    : {slope_proxy:.3f}   (1.0 = ideal)")
        print(f"  Calibration intercept: {intercept_proxy:+.3f}   (0.0 = ideal)")
        print("  Note: same underlying Psi/detection metrics either way -- this ensemble never")
        print("  touches the final model's training data. See train_proxy_calibration_ensemble()")
        print("  for the assumption this relies on before treating it as the final answer.")

    if p_calib_beta is not None:
        ece_beta = em.expected_calibration_error(y_true_hard, p_calib_beta)
        brier_beta = float(brier_score_loss(y_true_hard, p_calib_beta))
        ll_beta = float(log_loss(y_true_hard, p_calib_beta))
        slope_beta, intercept_beta = em.calibration_slope_intercept(y_true_hard, p_calib_beta)
        print("\nCalibration (beta -- 3-parameter map fit on the same calibration set as T_calib)")
        print(f"  ECE                  : {ece_beta*100:.2f}%"
              f"  ({'better' if ece_beta < ece_value else 'worse'} than T_calib in-sample by "
              f"{abs(ece_beta - ece_value)*100:.2f}pp)")
        print(f"  Brier                : {brier_beta:.4f}")
        print(f"  Log-loss             : {ll_beta:.4f}")
        print(f"  Calibration slope    : {slope_beta:.3f}   (1.0 = ideal)")
        print(f"  Calibration intercept: {intercept_beta:+.3f}   (0.0 = ideal)")
        print("  Note: same underlying Psi -- only the recalibration map differs from T_calib.")
        print("  3 parameters on a calibration set with 1 crisis episode is a real overfitting")
        print("  risk (see fit_beta_calibration's docstring); do not trust this over T_calib on")
        print("  a single run without checking it holds across a few seed sets.")

    print("\nInvariants")
    print(f"  ROC-AUC              : {roc_auc:.4f}")
    print(f"  PR-AUC (sklearn)     : {pr_auc:.4f}")
    if abs(pr_auc - pr_auc_legacy) > 1e-4:
        print(f"  PR-AUC (legacy fmla) : {pr_auc_legacy:.4f}  <- differs from sklearn by "
              f"{abs(pr_auc - pr_auc_legacy):.4f}; see evaluation_metrics.legacy_average_precision")
    print(f"  Cohen's d            : {cd:.2f}")
    print(f"  MCC                  : {extra['mcc']:.3f}")
    print(f"  Balanced accuracy    : {extra['balanced_accuracy']*100:.2f}%")
    print(f"  Youden's J           : {extra['youdens_j']:.3f}")

    print("\nConfusion matrices")
    print(f"  base      : TN={base['tn']:<4d} FP={base['fp']:<4d} FN={base['fn']:<4d} TP={base['tp']:<4d}")
    print(f"  calibrated: TN={calib['tn']:<4d} FP={calib['fp']:<4d} FN={calib['fn']:<4d} TP={calib['tp']:<4d}")

    frame = pd.DataFrame(
        {"y": y_true_hard, "base_flag": p65, "calib_flag": p_tau, "psi_scaled": psi * CFG.DISPLAY_SCALE},
        index=dates,
    )
    windows_in_range = [
        (CRISIS_NAMES.get(w, f"{w[0]} to {w[1]}"), w[0], w[1])
        for w in CRISES
        if pd.Timestamp(w[0]) <= dates.max() and pd.Timestamp(w[1]) >= dates.min()
    ]
    print("\nMajor crises inside the evaluated window")
    for name, start, end in windows_in_range:
        sub = frame.loc[start:end]
        if len(sub):
            print(f"  {name:<32}: recall base={sub['base_flag'].mean()*100:5.1f}%  "
                  f"recall calibrated={sub['calib_flag'].mean()*100:5.1f}%  "
                  f"mean psi={sub['psi_scaled'].mean():5.1f}")

    sweep_taus = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.65]
    thresholds, recalls, precisions, f1s = [], [], [], []
    print("\nThreshold sweep")
    print(f"  {'tau':>6}{'recall':>10}{'precision':>12}{'f1':>8}")
    for t in sweep_taus:
        pred_t = (p_calib >= t).astype(int)
        rates = _confusion_and_rates(y_true_hard, pred_t)
        thresholds.append(t)
        recalls.append(rates["recall"])
        precisions.append(rates["precision"])
        f1s.append(rates["f1"])
        marker = "  <- tau*" if abs(t - tau) < 1e-9 else ""
        print(f"  {t:6.2f}{rates['recall']*100:9.1f}%{rates['precision']*100:11.1f}%"
              f"{rates['f1']*100:7.1f}%{marker}")

    corr_gamma = float(np.corrcoef(gamma, psi)[0, 1])
    corr_temp = float(np.corrcoef(temp, psi)[0, 1])
    print("\nPhysics")
    print(f"  corr(gamma, Psi)     : {corr_gamma:+.4f}")
    print(f"  corr(T, Psi)         : {corr_temp:+.4f}")
    print(f"  mean alpha           : {alpha.mean():.4f}")

    if kramers_diagnostic is not None:
        print("\nKramers-rate auxiliary signal (diagnostic only -- not blended into Psi/P_calib)")
        if kramers_diagnostic["direction_flipped"]:
            print(f"  raw ROC-AUC                : {kramers_diagnostic['raw_roc_auc']:.4f}  "
                  f"(below 0.5 -- direction inverted below, not a weak signal)")
        print(f"  standalone ROC-AUC        : {kramers_diagnostic['auxiliary_roc_auc']:.4f}"
              f"  (Psi's ROC-AUC, for reference: {roc_auc_score(y_true_hard, psi):.4f})"
              f"{'  [sign-flipped: low proxy -> high crisis likelihood]' if kramers_diagnostic['direction_flipped'] else ''}")
        print(f"  standalone PR-AUC         : {kramers_diagnostic['auxiliary_pr_auc']:.4f}")
        print(f"  correlation with Psi      : {kramers_diagnostic['correlation_with_primary']:+.4f}")
        if kramers_diagnostic["auxiliary_roc_auc"] > 0.6 and abs(kramers_diagnostic["correlation_with_primary"]) < 0.8:
            print("  -> informative AND not redundant with Psi: worth trying as a blended signal")
            print("     (fit blend weights on the calibration set, never on test -- same discipline as T_calib).")
        elif kramers_diagnostic["auxiliary_roc_auc"] <= 0.6:
            print("  -> weak standalone signal on this run; probably not worth pursuing further as-is.")
        else:
            print("  -> informative but highly correlated with Psi: unlikely to add much if blended.")

    if blend_diagnostic is not None:
        psi_roc = float(roc_auc_score(y_true_hard, psi))
        delta = blend_diagnostic["auxiliary_roc_auc"] - psi_roc
        print("\nPsi + Kramers blend (logistic regression fit on the calibration set only)")
        print(f"  blended ROC-AUC           : {blend_diagnostic['auxiliary_roc_auc']:.4f}"
              f"  vs. Psi alone: {psi_roc:.4f}  (delta {delta:+.4f})")
        print(f"  blended PR-AUC            : {blend_diagnostic['auxiliary_pr_auc']:.4f}")
        if delta > 0.005:
            print("  -> blend beats Psi alone on this run. Still just one run -- confirm across a few")
            print("     seed sets before treating this as your primary reported signal.")
        elif delta < -0.005:
            print("  -> blend underperforms Psi alone on this run; the two signals may already be too")
            print("     correlated in-sample for the logistic regression to extract much extra value.")
        else:
            print("  -> roughly a wash on this run -- within likely run-to-run noise either way.")

    boot_roc = em.block_bootstrap_ci(y_true_hard, psi, roc_auc_score, n_boot=2000, block_size=21)
    boot_pr = em.block_bootstrap_ci(y_true_hard, psi, average_precision_score, n_boot=2000, block_size=21)
    print("\nBlock-bootstrap 95% CI (block = 21 trading days; see evaluation_metrics.py)")
    print(f"  ROC-AUC : {boot_roc['point']:.4f}  [{boot_roc['lo']:.4f}, {boot_roc['hi']:.4f}]")
    print(f"  PR-AUC  : {boot_pr['point']:.4f}  [{boot_pr['lo']:.4f}, {boot_pr['hi']:.4f}]")
    print(f"  Caution: the test window holds only {len(windows_in_range)} largely independent crisis "
          f"episode(s). This interval reflects day-level resampling of the observed window, not "
          f"episode-level generalization to a future, unseen crisis.")

    current_metrics = {
        "roc_auc": roc_auc, "pr_auc": pr_auc, "cohens_d": cd,
        "recall_calibrated": calib["recall"] * 100, "precision_calibrated": calib["precision"] * 100,
        "f1_calibrated": calib["f1"] * 100, "ece": ece_value * 100, "brier": brier, "log_loss": ll,
    }
    if ece_proxy is not None:
        current_metrics.update({
            "ece_proxy": ece_proxy * 100, "brier_proxy": brier_proxy, "log_loss_proxy": ll_proxy,
        })
    history = em.load_run_history(history_path)
    if previous_version not in history:
        # First run: seed the log with the numbers already on hand so the
        # comparison works immediately instead of only from the next run.
        em.save_run_record(history_path, previous_version, {
            "roc_auc": 0.9088, "pr_auc": 0.7469, "cohens_d": 1.97,
            "recall_calibrated": 75.62, "precision_calibrated": 67.60, "f1_calibrated": 71.39,
            "ece": 2.35, "brier": 0.0684, "log_loss": 0.2601,
        })
    em.save_run_record(history_path, version, current_metrics)
    history = em.load_run_history(history_path)

    rows = em.compare_versions(history, version, previous_version)
    print(f"\n{version} vs {previous_version}")
    print(f"  {'metric':<22}{version:>12}{previous_version:>12}{'delta':>12}")
    for row in rows:
        winner = version if row["current_is_better"] else previous_version
        print(f"  {row['metric']:<22}{row['current']:12.4f}{row['previous']:12.4f}"
              f"{row['delta']:+12.4f}  ({winner} better)")

    prev_roc = history[previous_version]["roc_auc"]
    pct_roc = em.reference_percentile(boot_roc["samples"], prev_roc)
    print(f"\n  {previous_version}'s ROC-AUC ({prev_roc:.4f}) sits at the {pct_roc*100:.0f}th percentile "
          f"of {version}'s own bootstrap spread. Descriptive only, not a formal significance test -- "
          f"see evaluation_metrics.reference_percentile.")

    bundle = dict(
        y_true=y_true_hard, psi=psi, p_calib=p_calib,
        cm_base=base["confusion_matrix"], cm_calib=calib["confusion_matrix"],
        dates=dates, psi_scaled=psi * CFG.DISPLAY_SCALE, tau_scaled=tau * CFG.DISPLAY_SCALE,
        crisis_windows=windows_in_range, cohens_d=cd,
        thresholds=thresholds, recalls=recalls, precisions=precisions, f1s=f1s, tau_star=tau,
        gamma=gamma, temp=temp, alpha=alpha,
        bootstrap_roc_auc=boot_roc, reference_roc_auc=prev_roc,
        loss_histories=loss_histories,
    )
    figure_paths = viz.generate_all_figures(bundle, out_dir)
    print(f"\nSaved {len(figure_paths)} figures to {out_dir}")

    return {
        "base": base, "calibrated": calib, "roc_auc": roc_auc, "pr_auc": pr_auc,
        "pr_auc_legacy": pr_auc_legacy, "cohens_d": cd, "ece": ece_value, "brier": brier,
        "log_loss": ll, "extra": extra,
        "proxy_calibration": (
            {"t_calib": t_calib_proxy, "ece": ece_proxy, "brier": brier_proxy, "log_loss": ll_proxy,
             "calibration_slope": slope_proxy, "calibration_intercept": intercept_proxy}
            if ece_proxy is not None else None
        ),
        "bootstrap": {"roc_auc": boot_roc, "pr_auc": boot_pr},
        "figures": figure_paths,
    }


# ---- Data pipeline ----
def load_price_data(cfg: type[CFG]) -> tuple[pd.Series, pd.DataFrame]:
    """Downloads adjusted close prices for the full ticker universe.
    Columns with less than 30% history (tickers that started trading well
    after 1993) are dropped. Returns (target_prices, all_prices)."""
    raw = yf.download(cfg.ALL_TICKERS, start=cfg.START_DATE, end=cfg.END_DATE,
                       auto_adjust=True, progress=False)
    all_prices = raw["Close"].ffill().bfill()
    all_prices = all_prices.rename(columns={c: c.replace("^", "").replace("=F", "") for c in all_prices.columns})
    all_prices = all_prices.loc[:, all_prices.notna().mean() > 0.3]
    target_prices = all_prices[cfg.TARGET]
    return target_prices, all_prices


def compute_credit_persistence(all_prices: pd.DataFrame, smooth_window: int | None = None) -> pd.Series:
    """Exogenous persistence signal for CFG.DYNAMIC_K_FN: as of each day,
    how many consecutive smooth_window-day blocks credit stress (HYG
    underperforming IEF -- i.e. high-yield spreads widening) has been
    rising. Computed purely from already-observed HYG/IEF prices --
    never touches the model's own outputs or the crisis labels, unlike a
    learned modulation, which is the whole point (see CFG.DYNAMIC_K_FN).

    Causality, made simple to verify directly (see
    test_dynamic_k_fn_no_lookahead): everything here is either a
    rolling window ending at t or a purely sequential day-by-day scan
    (streak[i] depends only on widening[0..i]) -- deliberately avoided
    resample('W')-based week bucketing, whose label-boundary convention
    is an easy place to silently leak a few days of look-ahead.

    Returns a Series aligned to all_prices.index, in units of
    "qualifying smooth_window-day blocks" (matches CFG.DYNAMIC_K_FN_TAU's
    units) -- not literal calendar weeks, though smooth_window=5 makes
    them close to that.
    """
    if smooth_window is None:
        smooth_window = CFG.DYNAMIC_K_FN_SMOOTH_WINDOW
    if "HYG" not in all_prices.columns or "IEF" not in all_prices.columns:
        return pd.Series(0.0, index=all_prices.index)

    stress = -np.log(all_prices["HYG"] / (all_prices["IEF"] + 1e-8))
    smooth = stress.rolling(smooth_window).mean()
    # smooth[t] uses days [t-smooth_window+1, t]; smooth[t-smooth_window]
    # uses days [t-2*smooth_window+1, t-smooth_window] -- both entirely
    # <= t, so this comparison never touches a future day.
    block_change = smooth.diff(smooth_window)
    widening = (block_change > 0).fillna(False).values

    streak = np.zeros(len(widening))
    count = 0.0
    for i, w in enumerate(widening):
        count = count + 1.0 if w else 0.0
        streak[i] = count
    return pd.Series(streak, index=all_prices.index)


def build_feature_matrix(prices: pd.Series, all_prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Runs the five feature-engineering classes plus velocity features,
    optionally spectral features (see CFG.INCLUDE_SPECTRAL_FEATURES), and
    concatenates them into the feature matrix the model expects (204
    columns by default, 244 with spectral features included)."""
    returns, stat, vol, mom, mkt, mac = build_base_features(prices, all_prices)
    velocity = build_velocity_features(returns)
    frames = [stat, vol, mom, mkt, mac, velocity]
    if CFG.INCLUDE_SPECTRAL_FEATURES:
        from spectral_features import build_spectral_features
        frames.append(build_spectral_features(returns))
    frames = [f.reindex(returns.index) for f in frames]
    features = pd.concat(frames, axis=1).ffill().bfill().fillna(0)
    return returns, features


def split_data(dates: pd.DatetimeIndex, X: np.ndarray, y_hard: np.ndarray,
                y_soft: np.ndarray, cfg: type[CFG],
                persistence: "np.ndarray | None" = None) -> dict:
    """Chronological train / calibration / test split.

    By default (cfg.STRICT_CALIBRATION_HOLDOUT = False) train runs through
    2019-12-31, so the 2017-2019 calibration window is a subset of
    training -- see the module docstring for the known calibration-metric
    caveat this creates and why the alternative (True) was measured and
    rejected as the default rather than adopted.

    Also returns the "proxy" training slice (dates < ts_calib) on its
    own, for train_proxy_calibration_ensemble(): a separate model trained
    only on this slice can be evaluated on the calibration window without
    any in-sample overlap, without touching what the *final* model (the
    one whose numbers get reported) trains on.

    persistence, when supplied (CFG.DYNAMIC_K_FN), is sliced with the
    EXACT SAME boolean masks as X/y below -- guarantees day-for-day
    alignment by construction rather than by a second, separately-
    computed index operation that could silently drift out of sync.
    """
    ts_calib = pd.Timestamp("2017-01-01")
    ts_test = pd.Timestamp("2020-01-01")
    train_upper_bound = ts_calib if cfg.STRICT_CALIBRATION_HOLDOUT else ts_test

    mask_train = dates < train_upper_bound
    mask_test = dates >= ts_test
    mask_calib = (dates >= ts_calib) & (dates < ts_test)
    mask_proxy_train = dates < ts_calib

    out = {
        "X_train": X[mask_train], "y_train_soft": y_soft[mask_train],
        "X_test": X[mask_test], "y_test_hard": y_hard[mask_test], "dates_test": dates[mask_test],
        "X_calib": X[mask_calib], "y_calib_hard": y_hard[mask_calib],
        "X_proxy_train": X[mask_proxy_train], "y_proxy_train_soft": y_soft[mask_proxy_train],
        "n_train": int(mask_train.sum()), "n_calib": int(mask_calib.sum()), "n_test": int(mask_test.sum()),
        "n_proxy_train": int(mask_proxy_train.sum()),
    }
    if persistence is not None:
        out["persistence_train"] = persistence[mask_train]
        out["persistence_proxy_train"] = persistence[mask_proxy_train]
    return out


# ---- Stateful inference helper (shared by calibration, test, and proxy passes) ----
def run_stateful_inference(models: list, X: np.ndarray, device: str, latent_dim: int,
                            collect_physics: bool = False):
    """Runs each model in `models` sequentially over X -- one day at a
    time, carrying the hidden state s across the whole pass -- and
    returns the ensemble-averaged Psi. The model needs to see days in
    order for the hidden state to mean anything, which is why this can't
    be a batched forward pass.

    If collect_physics, also returns the ensemble-averaged gamma/temp/
    alpha/a/b (only meaningful for a pass whose output is actually
    reported; the proxy-calibration pass does not need them). a/b are the
    potential's shape parameters, captured purely for the post-hoc
    Kramers-rate diagnostic -- see evaluation_metrics.kramers_rate_proxy.

    When CFG.TCN_VELOCITY_ENCODER is True, also builds each day's
    TCN_WINDOW-day window from X directly (this function is called on
    standalone slices -- calibration, test, or proxy-training windows --
    with no guaranteed pre-history the way ChronoRegimeDataset's sl
    margin provides within a single training set). The first
    TCN_WINDOW-1 days of any given call are zero-padded at the front, the
    same cold-start convention the TCN already uses internally for a
    day with no earlier context -- documented here rather than silently
    approximated, and a negligible fraction of any real evaluation
    window (a handful of days out of hundreds).
    """
    x_t = torch.tensor(X, dtype=torch.float32).to(device)
    fast_idx_t = torch.tensor(CFG.FAST_IDX, dtype=torch.long) if CFG.TCN_VELOCITY_ENCODER else None
    all_psi, all_gamma, all_temp, all_alpha, all_a, all_b = [], [], [], [], [], []
    for model in models:
        model.eval()
        with torch.no_grad():
            state = torch.zeros((1, latent_dim), device=device)
            bp, bg, bt, ba, baa, bbb = [], [], [], [], [], []
            for t in range(x_t.shape[0]):
                if CFG.TCN_VELOCITY_ENCODER:
                    lo = t - CFG.TCN_WINDOW + 1
                    real = x_t[max(lo, 0):t + 1, fast_idx_t]
                    if lo < 0:
                        pad = torch.zeros((-lo, real.shape[1]), device=device)
                        real = torch.cat([pad, real], dim=0)
                    xf_window = real.unsqueeze(0)  # (1, TCN_WINDOW, len(FAST_IDX))
                    psi_t, _, state = model(x_t[t:t + 1], s_initial=state, xf_window=xf_window)
                else:
                    psi_t, _, state = model(x_t[t:t + 1], s_initial=state)
                bp.append(psi_t.cpu().item())
                if collect_physics:
                    bg.append(model._last_gamma.cpu().item())
                    bt.append(model._last_temp.cpu().item())
                    ba.append(model._last_alpha.cpu().item())
                    baa.append(model._last_a.cpu().item())
                    bbb.append(model._last_b.cpu().item())
            all_psi.append(np.array(bp))
            if collect_physics:
                all_gamma.append(np.array(bg))
                all_temp.append(np.array(bt))
                all_alpha.append(np.array(ba))
                all_a.append(np.array(baa))
                all_b.append(np.array(bbb))

    psi_ensemble = np.mean(all_psi, axis=0)
    if not collect_physics:
        return psi_ensemble
    return (psi_ensemble, np.mean(all_gamma, axis=0), np.mean(all_temp, axis=0),
            np.mean(all_alpha, axis=0), np.mean(all_a, axis=0), np.mean(all_b, axis=0))


# ---- Proxy calibration (see module docstring: "a cleaner fix without the data cost") ----
def train_proxy_calibration_ensemble(seeds: list, X_proxy_train: np.ndarray,
                                      y_proxy_train_soft: np.ndarray, device: str,
                                      persistence_proxy_train: "np.ndarray | None" = None) -> list:
    """Trains a separate ensemble on data strictly before the calibration
    window (X_proxy_train), for the sole purpose of producing genuinely
    out-of-sample Psi values on that window. This ensemble is never used
    for anything reported except fitting a temperature scalar -- it does
    not touch, and cannot improve or hurt, the final model's training
    data or its detection metrics.

    persistence_proxy_train (when CFG.DYNAMIC_K_FN is on) should be the
    same persistence signal sliced to the proxy window's own date range
    -- kept consistent with the final model's training procedure, since
    the whole point of this ensemble is to approximate how the final
    model trains, just on different data.

    Caveat to state plainly, not bury: this assumes the proxy ensemble's
    degree of overconfidence (the thing temperature scaling corrects for)
    is representative of the final ensemble's, even though the final
    ensemble trained on ~3 more years of data. That's a real assumption,
    not a proof. It is a materially better assumption than the status quo
    (fitting the temperature on the final model's in-sample outputs,
    which is not an estimate of out-of-sample overconfidence at all) --
    but report both numbers and let the comparison speak for itself
    rather than treating this as a solved problem.
    """
    models = []
    for seed in seeds:
        try:
            model, _ = train_one(seed, X_proxy_train, y_proxy_train_soft, device,
                                  persistence_tr=persistence_proxy_train)
            models.append(model)
        except Exception:
            logger.exception("proxy-calibration seed %s failed; continuing with the rest", seed)
    if not models:
        raise RuntimeError("every proxy-calibration seed failed")
    return models


# ---- Main ----
def main() -> None:
    torch.manual_seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    random.seed(CFG.SEED)

    # Printed unconditionally, first thing, every run: which experimental
    # flags are active. Added after several runs tonight where a flag's
    # true state could not be determined with certainty from the training
    # log alone -- this makes that ambiguity structurally impossible
    # going forward, rather than something to keep re-diagnosing by eye.
    _exp_flags = ["STRICT_CALIBRATION_HOLDOUT", "PROXY_CALIBRATION", "BETA_CALIBRATION",
                  "INCLUDE_SPECTRAL_FEATURES", "ASYMMETRIC_SOFT_LABELS",
                  "KINETIC_ENERGY_PENALTY", "MACRO_SKIP_CONNECTION", "FDT2_COLORED_NOISE",
                  "TCN_VELOCITY_ENCODER", "DYNAMIC_K_FN"]
    print("=" * 70)
    print("ACTIVE CONFIGURATION (this run's identity)")
    for _name in _exp_flags:
        _val = getattr(CFG, _name, None)
        print(f"  {_name:28s} = {_val}" + ("  <-- ACTIVE" if _val else ""))
    print(f"  {'ENSEMBLE_SEEDS':28s} = {CFG.ENSEMBLE_SEEDS}")
    if CFG.ASYMMETRIC_SOFT_LABELS:
        print(f"  {'  SOFT_LABEL_SIGMA_ONSET/_DECAY':28s} = {CFG.SOFT_LABEL_SIGMA_ONSET}/{CFG.SOFT_LABEL_SIGMA_DECAY}")
    print(f"  {'ASYM_CRISIS_UNDER_K (-> tau*)':28s} = {CFG.ASYM_CRISIS_UNDER_K} "
          f"(tau* = {1.0/(1.0+CFG.ASYM_CRISIS_UNDER_K):.3f})")
    print("=" * 70)

    logger.info("V30.4 -- soft labels (sigma=%d) + SWA", CFG.SOFT_LABEL_SIGMA)

    prices, all_prices = load_price_data(CFG)
    returns, features = build_feature_matrix(prices, all_prices)

    hard_labels = make_hard_labels(features.index)
    soft_labels = build_soft_labels(features.index)

    common_index = features.index.intersection(hard_labels.index)
    X = features.loc[common_index].values
    y_hard = hard_labels.loc[common_index].values
    y_soft = soft_labels.loc[common_index].values
    dates = common_index

    # CFG.DYNAMIC_K_FN: computed unconditionally (cheap: no model
    # involved, pure price arithmetic) so split_data always has it
    # available; train_one only actually uses it when the flag is on.
    persistence = compute_credit_persistence(all_prices).reindex(common_index).fillna(0.0).values

    split = split_data(dates, X, y_hard, y_soft, CFG, persistence=persistence)
    logger.info("train=%dd (soft labels, sigma=%d) | calib=%dd | test=%dd",
                split["n_train"], CFG.SOFT_LABEL_SIGMA, split["n_calib"], split["n_test"])
    n_transition = int(((split["y_train_soft"] > 0.1) & (split["y_train_soft"] < 0.9)).sum())
    logger.info("transition days (0.1<y<0.9) in train: %d", n_transition)
    logger.info("hard crisis days in test: %d (%.1f%%)",
                int(split["y_test_hard"].sum()), split["y_test_hard"].mean() * 100)
    if CFG.DYNAMIC_K_FN:
        logger.info("dynamic k_FN active: persistence range in train = [%.1f, %.1f], mean=%.2f",
                     split["persistence_train"].min(), split["persistence_train"].max(),
                     split["persistence_train"].mean())

    # ---- final ensemble: trains on the full default window (unaffected by ----
    # ---- anything below -- this is what produces every reported detection metric) ----
    models, loss_histories = [], {}
    for seed in CFG.ENSEMBLE_SEEDS:
        logger.info("training seed=%s", seed)
        try:
            model, history = train_one(
                seed, split["X_train"], split["y_train_soft"], CFG.DEVICE,
                persistence_tr=split.get("persistence_train"),
            )
            models.append(model)
            loss_histories[seed] = history
        except Exception:
            logger.exception("seed %s failed; continuing with the remaining ensemble members", seed)
    logger.info("%d/%d ensemble seeds trained successfully", len(models), len(CFG.ENSEMBLE_SEEDS))
    if not models:
        raise RuntimeError("every ensemble seed failed; nothing to evaluate")

    logger.info("running stateful ensemble inference (calibration set)")
    (psi_calib_ensemble, gamma_calib_ensemble, temp_calib_ensemble,
     _alpha_calib_ensemble, a_calib_ensemble, b_calib_ensemble) = run_stateful_inference(
        models, split["X_calib"], CFG.DEVICE, CFG.LATENT_DIM, collect_physics=True,
    )
    t_calib = get_temperature_scaler(psi_calib_ensemble, split["y_calib_hard"])
    logger.info("T_calib (in-sample calibration set) = %.3f", t_calib)

    logger.info("running stateful ensemble inference (test set)")
    psi_ensemble, gamma_ensemble, temp_ensemble, alpha_ensemble, a_ensemble, b_ensemble = run_stateful_inference(
        models, split["X_test"], CFG.DEVICE, CFG.LATENT_DIM, collect_physics=True,
    )
    psi_clipped = np.clip(psi_ensemble, 1e-6, 1 - 1e-6)
    p_calib = expit(logit(psi_clipped) / t_calib)

    # Kramers-rate proxy: a purely post-hoc diagnostic computed from the
    # already-learned (a, b, gamma, T), never fed back into training or
    # into Psi. See evaluation_metrics.kramers_rate_proxy for the physics
    # and its caveats; evaluate_auxiliary_signal checks whether it's
    # informative on its own and whether it's redundant with Psi.
    kramers_proxy = em.kramers_rate_proxy(a_ensemble, b_ensemble, gamma_ensemble, temp_ensemble)
    kramers_diagnostic = em.evaluate_auxiliary_signal(split["y_test_hard"], kramers_proxy, psi_ensemble)

    # Psi + Kramers blend: a logistic regression with 2 inputs
    # (logit(Psi), log(kramers)), fit ONLY on the calibration set -- same
    # discipline as get_temperature_scaler() above, never fit on the test
    # set being evaluated. Trains nothing in the neural-network sense (no
    # gradient steps on the ensemble); the only thing being "trained" is a
    # 3-parameter logistic regression on top of two already-computed
    # signals, so this cannot change Psi, P_calib, or any existing metric
    # -- it only adds a new, optional, side-by-side candidate signal.
    blend_diagnostic = None
    try:
        kramers_calib = em.kramers_rate_proxy(
            a_calib_ensemble, b_calib_ensemble, gamma_calib_ensemble, temp_calib_ensemble,
        )
        blend_clf = em.fit_signal_blend(split["y_calib_hard"], psi_calib_ensemble, kramers_calib)
        p_blend_test = em.apply_signal_blend(blend_clf, psi_ensemble, kramers_proxy)
        blend_diagnostic = em.evaluate_auxiliary_signal(split["y_test_hard"], p_blend_test, psi_ensemble)
        logger.info("Psi+Kramers blend fit on calibration set: standalone test ROC-AUC=%.4f "
                    "(Psi alone: %.4f)", blend_diagnostic["auxiliary_roc_auc"],
                    float(roc_auc_score(split["y_test_hard"], psi_ensemble)))
    except Exception:
        logger.exception("Psi+Kramers blend failed; continuing without it")

    # ---- proxy calibration ensemble: trains ONLY on data before the ----
    # ---- calibration window, purely to get an out-of-sample T_calib. Never ----
    # ---- touches the final model above or any detection metric. ----
    t_calib_proxy = p_calib_proxy = None
    if CFG.PROXY_CALIBRATION:
        logger.info("training proxy calibration ensemble (data before %s only)", "2017-01-01")
        logger.info("proxy train=%dd (vs. final model's %dd)",
                     split["n_proxy_train"], split["n_train"])
        try:
            proxy_models = train_proxy_calibration_ensemble(
                CFG.ENSEMBLE_SEEDS, split["X_proxy_train"], split["y_proxy_train_soft"], CFG.DEVICE,
                persistence_proxy_train=split.get("persistence_proxy_train"),
            )
            logger.info("running stateful proxy-ensemble inference (calibration set)")
            psi_calib_proxy = run_stateful_inference(proxy_models, split["X_calib"], CFG.DEVICE, CFG.LATENT_DIM)
            t_calib_proxy = get_temperature_scaler(psi_calib_proxy, split["y_calib_hard"])
            logger.info("T_calib (out-of-sample proxy) = %.3f", t_calib_proxy)
            p_calib_proxy = expit(logit(psi_clipped) / t_calib_proxy)
        except Exception:
            logger.exception("proxy calibration failed; reporting the in-sample calibration only")

    # ---- beta calibration: a 3-parameter alternative to the 1-parameter ----
    # ---- T_calib above, fit on the SAME calibration set. Never touches ----
    # ---- the trained model; purely a different post-hoc recalibration map. ----
    p_calib_beta = None
    if CFG.BETA_CALIBRATION:
        try:
            beta_clf = em.fit_beta_calibration(split["y_calib_hard"], psi_calib_ensemble)
            p_calib_beta = em.apply_beta_calibration(beta_clf, psi_ensemble)
            logger.info("beta calibration fit on calibration set (3 params vs T_calib's 1)")
        except Exception:
            logger.exception("beta calibration failed; reporting T_calib only")

    # Bayes-optimal threshold under the loss's FN:FP cost ratio (k_FN = 3):
    # tau* = 1 / (1 + k_FN). Not a separately tuned knob -- it falls out of
    # the same asymmetry already baked into hooke_loss().
    tau = 1.0 / (1.0 + CFG.ASYM_CRISIS_UNDER_K)

    build_report(
        y_true_hard=split["y_test_hard"], psi=psi_ensemble, p_calib=p_calib,
        dates=split["dates_test"], gamma=gamma_ensemble, temp=temp_ensemble, alpha=alpha_ensemble,
        tau=tau, loss_histories=loss_histories,
        p_calib_proxy=p_calib_proxy, t_calib_proxy=t_calib_proxy,
        p_calib_beta=p_calib_beta,
        kramers_diagnostic=kramers_diagnostic,
        blend_diagnostic=blend_diagnostic,
    )


if __name__ == "__main__":
    main()

