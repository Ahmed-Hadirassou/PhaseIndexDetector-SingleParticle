"""
tilt_ablation.py

Four-branch ablation on the symmetry-breaking term c of the
Ginzburg-Landau potential, testing where any gain over the current
model comes from:

  B1 "v2"            c = 2*tanh(w_c . h)                  (exactly v2)
  B2 "horizon"       c = 2*tanh(w_c . h'),  h' also sees M_t as an input,
                     with no imposed structure
  B3 "physics"       c = 2*tanh(g * z_M)                  (network's c ignored)
  B4 "coupled"       c = 2*tanh(g * z_M + w_c . h)

Three design decisions, all aimed at making the four numbers actually
comparable rather than merely different:

1. THE BRANCHES ARE NESTED. B4 reduces to B1 at g = 0 and to B3 when
   the network's c output is 0. B1, B3 and B4 all have the identical
   parameter count and the identical reachable range c in [-2, 2],
   because the tilt enters the same 2*tanh(.) that v2 already uses.
   Without this, a branch that loses could be losing on range or on
   capacity rather than on the hypothesis under test. (B2 alone carries
   32 extra weights, from the one extra input column: unavoidable,
   since the extra input is the thing being tested.)

2. NO NEW TUNED HYPERPARAMETERS. The earlier sketch of this experiment
   carried a fixed amplitude c0, a fixed slope beta and a hardcoded
   sign. All three are removed here: a single learnable scalar g
   absorbs amplitude, slope and sign together. This matters beyond
   tidiness. Nothing in the potential fixes which basin is the crisis
   basin, because V(z) = a|z|^4 - b|z|^2 + c*mean(z) is read out by a
   LEARNED head; a hardcoded sign is an assertion about an orientation
   the model is free to choose, and if it is backwards then B3 fails for
   a reason that has nothing to do with Chiarella. A learnable g keeps
   the saturating functional form, which is the whole point of the
   inductive bias, and lets orientation be discovered.

3. tau IS FIXED AT 200 A PRIORI, not tuned. It is chosen to match the
   horizon of the moving-average baseline the experiment exists to
   explain. Sweeping it over six crises would buy a better number at the
   cost of the only thing this experiment is for.

The trend signal follows the Chiarella form used in Kurth-Bouchaud
rather than a raw price-to-moving-average distance: an EMA of past log
returns, which has no rectangular-window discontinuity at t - 200.

    M_t = (1 - lambda) M_{t-1} + lambda r_t,    lambda = 2 / (tau + 1)

M_t is appended as one extra column of the SLOW feature block. It is
therefore z-scored by the model's own ZScoreNorm, which is fitted
inside train_one on the training window only. That is deliberate: it
reuses the existing causal normalizer instead of adding a second,
separately-written one that could leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG, ParametricBistablePotential

TREND_TAU = 200                       # fixed a priori, never swept
MODES = ("v2", "horizon", "physics", "coupled")

# Set by run_tilt_ablation before each branch; read at construction time
# by TiltedBistablePotential, since PhaseIndexDetector builds its own
# potential internally and takes no argument for this.
TILT_MODE = "v2"


def compute_trend_signal(prices: pd.Series, tau: int = TREND_TAU) -> pd.Series:
    """Chiarella chartist variable: EMA of past log returns. Uses only
    information up to and including t at every t (pandas ewm is causal),
    so this is safe to compute once over the full sample and slice per
    fold. No standardization here: the model's own ZScoreNorm, fitted on
    each fold's training window, supplies z(M_t)."""
    r = np.log(prices / prices.shift(1))
    lam = 2.0 / (tau + 1.0)
    return r.ewm(alpha=lam, adjust=False).mean().rename("M_t")


class TiltedBistablePotential(ParametricBistablePotential):
    """Drop-in subclass of the real potential. Inherits energy(),
    gradient() and forward() untouched, so the Verlet integrator's
    contract (potential(x_slow, z, x_macro_skip, vol_signal) -> (energy,
    grad); potential.get_params(...) -> (a, b, c)) is preserved exactly.
    Only get_params is overridden.

    x_slow arrives with the M_t column appended and already normalized,
    i.e. width slow_dim = 81, of which the first 80 are the original v2
    slow block in the original order.
    """

    def __init__(self, slow_dim: int, latent_dim: int, hidden: int = 32,
                 macro_skip_dim: int = 0, mode: str | None = None):
        mode = mode if mode is not None else TILT_MODE
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._n_base = slow_dim - 1                       # 80 original slow features
        net_in = slow_dim if mode == "horizon" else self._n_base
        # Build the parent with the width param_net should actually have,
        # so that in every mode except "horizon" the Linear layers are
        # created with exactly v2's shapes. With the same torch seed, and
        # because ZScoreNorm holds only buffers and consumes no RNG, this
        # makes branch "v2" bit-identical to the deployed model.
        super().__init__(net_in, latent_dim, hidden, macro_skip_dim)
        self.mode = mode
        if mode in ("physics", "coupled"):
            # Single learnable scalar: amplitude, slope and sign at once.
            self.g = torch.nn.Parameter(torch.tensor(1.0))

    def get_params(self, x_slow, x_macro_skip=None, vol_signal=None):
        if self.mode == "horizon":
            net_in = x_slow
        else:
            net_in = x_slow[:, : self._n_base]
        raw = self.param_net(net_in)
        if self.macro_skip_dim > 0 and x_macro_skip is not None:
            raw = raw + self.macro_skip_net(x_macro_skip)

        a = F.softplus(raw[:, 0:1]) + 0.1
        b = F.softplus(raw[:, 1:2]) + 0.1

        if self.mode in ("v2", "horizon"):
            c = torch.tanh(raw[:, 2:3]) * 2.0
        else:
            z_m = x_slow[:, self._n_base : self._n_base + 1]   # normalized M_t
            if self.mode == "physics":
                c = torch.tanh(self.g * z_m) * 2.0
            else:  # coupled: strict generalization of both v2 and physics
                c = torch.tanh(self.g * z_m + raw[:, 2:3]) * 2.0
        return a, b, c


def install(mode: str) -> None:
    """Point the model factory at the tilted potential in `mode`.
    PhaseIndexDetector constructs ParametricBistablePotential by name
    from its own module namespace, so rebinding that name is what makes
    the swap take effect inside train_one without editing the file."""
    global TILT_MODE
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    TILT_MODE = mode
    mvsl.ParametricBistablePotential = TiltedBistablePotential


def restore() -> None:
    """Undo install(); the original class is put back."""
    mvsl.ParametricBistablePotential = ParametricBistablePotential


def append_trend_column(X: np.ndarray, M: np.ndarray):
    """Append M_t as the last column of X and return (X', slow_idx',
    slow_dim'). Appending at the END keeps every existing absolute
    column index in CFG.SLOW_IDX / CFG.FAST_IDX / CFG.MACRO_SKIP_IDX
    valid, and putting the new index LAST inside SLOW_IDX is what makes
    the M_t column land at position self._n_base within x_slow."""
    X2 = np.concatenate([X, M.reshape(-1, 1).astype(np.float32)], axis=1)
    new_col = X.shape[1]
    slow_idx = list(CFG.SLOW_IDX) + [new_col]
    return X2, slow_idx, len(slow_idx)
