"""
Statistical evaluation utilities for the PhaseIndex crisis-detection model.

Everything here operates on model outputs that have already been computed
(true labels, raw scores, calibrated probabilities). Nothing in this module
touches the model, the features, or the training loop, so importing it
cannot change a single number in the base report.

Two things this module adds on top of the base report:

1. Point-estimate diagnostics the base report does not compute: MCC,
   balanced accuracy, Youden's J, a Brier Skill Score against the naive
   "always predict the base rate" baseline, and a calibration slope /
   intercept (a well-calibrated model has slope ~= 1, intercept ~= 0).

2. A block bootstrap for confidence intervals. A plain i.i.d. bootstrap on
   daily observations would treat each trading day as independent evidence,
   which overstates precision here: consecutive days inside the same crisis
   episode are strongly autocorrelated (if today is a COVID panic day,
   tomorrow almost certainly is too). Resampling contiguous blocks instead
   of single days is a standard fix for time-series bootstrap and gives a
   more honest (wider) interval. It does not fully solve the deeper issue --
   see the note on `block_bootstrap_ci` -- but it is a meaningfully better
   approximation than an i.i.d. bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Point-estimate metrics
# ---------------------------------------------------------------------------

def cohens_d(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Standardized mean separation between the crisis and stable score
    distributions. Same definition already used in the base report; kept
    here too so bootstrap resamples can call a single metric function."""
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    pooled_std = np.sqrt((pos.var() + neg.var()) / 2.0)
    return float((pos.mean() - neg.mean()) / (pooled_std + 1e-8))


def youdens_j(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sensitivity + specificity - 1. The threshold-dependent counterpart
    to ROC-AUC; 0 = no better than chance at this operating point, 1 = perfect."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return float(sensitivity + specificity - 1.0)


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Skill relative to predicting the base rate every day (the
    'climatology' baseline in forecast-verification terminology).
    0 = no better than that baseline, 1 = perfect, negative = worse."""
    base_rate = y_true.mean()
    brier_model = brier_score_loss(y_true, y_prob)
    brier_climatology = brier_score_loss(y_true, np.full_like(y_prob, base_rate, dtype=float))
    return float(1.0 - brier_model / (brier_climatology + 1e-12))


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Fits y ~ intercept + slope * logit(p) by unregularized logistic
    regression. Slope < 1 means predictions are too extreme (overconfident);
    slope > 1 means too conservative. Intercept != 0 means a systematic
    over/under-prediction of the base rate."""
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    # C is deliberately huge (near-zero regularization) rather than using
    # penalty=None directly: that argument's spelling has changed across
    # sklearn versions, while a large C reproduces an unregularized fit
    # on every version.
    clf = LogisticRegression(C=1e10, max_iter=2000)
    clf.fit(logit_p, y_true)
    return float(clf.coef_[0, 0]), float(clf.intercept_[0])


def point_estimates(y_true: np.ndarray, scores: np.ndarray, y_prob: np.ndarray,
                     y_pred: np.ndarray) -> dict:
    """One-shot bundle of every 'extra' indicator, all at a given operating
    point (y_pred = binarized prediction at whatever threshold is being
    evaluated; scores = continuous psi; y_prob = calibrated probability)."""
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "cohens_d": cohens_d(scores, y_true),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "youdens_j": youdens_j(y_true, y_pred),
        "brier_skill_score": brier_skill_score(y_true, y_prob),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width-bin ECE: bin by predicted probability, weight
    each bin's |mean predicted - mean observed| by its share of samples.
    Bin assignment uses the same `np.digitize` scheme as the reliability
    diagram in visualization.py, so the printed ECE number and the plot
    it describes are always reading the same bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n = len(y_true)
    error = 0.0
    for b in range(n_bins):
        m = bin_ids == b
        if not m.any():
            continue
        error += abs(y_prob[m].mean() - y_true[m].mean()) * (m.sum() / n)
    return float(error)


def legacy_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Reproduces the hand-rolled PR-AUC formula from the original V30.3 /
    V30.4 script, kept only so build_report() can show a one-time diagnostic
    against sklearn.metrics.average_precision_score.

    The two are the same definition of Average Precision, but they resolve
    tied scores differently: this version ranks tied samples via
    `np.argsort`, which (with NumPy's default non-stable quicksort) breaks
    ties in an order that has no relationship to any real decision
    threshold. sklearn groups tied scores and evaluates them at one shared
    threshold, which is what a real classifier actually does when several
    days get an identical score. On data with many exact ties this can move
    PR-AUC by several thousandths -- comparable in size to the V30.4-vs-V30.3
    deltas being compared in the report. Use the sklearn number as the one
    that goes in the paper; this function exists to quantify the gap on
    your real predictions, not to replace the sklearn computation.
    """
    order = np.argsort(-scores)
    ys = y_true[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    return float(np.sum(
        np.diff(np.concatenate([[0], tps / (y_true.sum() + 1e-8)]))
        * (tps / (tps + fps + 1e-8))
    ))


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Overlapping moving-block bootstrap: draw contiguous blocks of
    `block_size` consecutive indices, with replacement, and concatenate
    until there are at least n indices, then truncate to exactly n."""
    n_blocks = int(np.ceil(n / block_size))
    max_start = max(n - block_size, 1)
    starts = rng.integers(0, max_start, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])
    idx = np.clip(idx, 0, n - 1)
    return idx[:n]


def block_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    block_size: int = 21,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile block-bootstrap confidence interval for metric_fn(y, scores).

    block_size defaults to 21 trading days (~1 calendar month), long enough
    to keep most of a single crisis episode's autocorrelation inside one
    block rather than splitting it across independent draws.

    Caveat this does not fix: the test window used here contains only two
    largely independent crisis *episodes* (COVID 2020, the 2022 drawdown).
    Block resampling gives an honest estimate of "uncertainty from
    resampling the days in this particular window" -- it is not, and cannot
    be, an estimate of "how this model would perform on a crisis episode it
    has never seen." That second question needs more historical episodes
    than this dataset has, full stop; no resampling scheme manufactures
    them. Report both numbers, and say so.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = _block_bootstrap_indices(n, block_size, rng)
        yb, sb = y_true[idx], scores[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue  # degenerate resample (no positives, or no negatives)
        samples[b] = metric_fn(yb, sb)
    samples = samples[~np.isnan(samples)]
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(metric_fn(y_true, scores)),
        "lo": float(lo),
        "hi": float(hi),
        "samples": samples,
        "n_valid_boot": int(len(samples)),
        "block_size": block_size,
    }


def reference_percentile(bootstrap_samples: np.ndarray, reference_value: float) -> float:
    """Fraction of this version's bootstrap draws that fall at or below a
    reference value (typically a prior version's point estimate). This is
    descriptive, not a formal hypothesis test: we only have V30.3's summary
    statistics, not its raw per-day predictions, so a proper paired test
    (e.g. DeLong) isn't available. Use it to see whether the prior version's
    number sits comfortably inside this version's resampling spread, or out
    in the tail."""
    return float(np.mean(bootstrap_samples <= reference_value))


# ---------------------------------------------------------------------------
# Run history -- small append-only log so each new version is automatically
# compared to its predecessor instead of hand-copying numbers into the code
# ---------------------------------------------------------------------------

def load_run_history(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_run_record(path: str | Path, version: str, metrics: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history = load_run_history(path)
    history[version] = metrics
    path.write_text(json.dumps(history, indent=2, sort_keys=True))


def compare_versions(
    history: dict,
    current: str,
    previous: str,
    lower_is_better: Sequence[str] = ("ece", "brier", "log_loss"),
) -> list[dict]:
    """Returns one row per metric present in both versions: current value,
    previous value, delta, and whether the delta favors `current`."""
    if previous not in history or current not in history:
        return []
    rows = []
    for key, v_now in history[current].items():
        if key not in history[previous]:
            continue
        v_prev = history[previous][key]
        delta = v_now - v_prev
        lb = key in lower_is_better
        current_is_better = (delta < 0) if lb else (delta > 0)
        rows.append({
            "metric": key, "current": v_now, "previous": v_prev,
            "delta": delta, "current_is_better": current_is_better,
        })
    return rows


# ---------------------------------------------------------------------------
# Kramers-rate proxy -- a physics-derived diagnostic, computed post-hoc from
# the model's own learned potential/dynamics parameters. Does not feed back
# into training or into Psi; purely a check for complementary signal.
# ---------------------------------------------------------------------------

def kramers_rate_proxy(a: np.ndarray, b: np.ndarray, gamma: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """Analytic transition-rate proxy from Kramers escape-rate theory,
    evaluated on the model's own per-day potential V(z) = a*z^4 - b*z^2
    (+ c*z tilt, dropped here -- see caveat).

    For the untilted symmetric double well, the minima sit at
    z = +-sqrt(b / 2a) and the barrier top at z = 0, giving an exact
    barrier height:
        delta_V = b^2 / (4a)
    The Kramers rate in the moderate-to-high-friction (Smoluchowski)
    limit scales as:
        rate ~ gamma^-1 * exp(-delta_V / T)

    This is a RANK-preserving proxy, not a calibrated absolute rate: the
    prefactor here drops the well/barrier curvature terms (omega_0,
    omega_b) and the friction-regime crossover (gamma spans a wide range
    in this model -- roughly 0 to 0.95 across training logs -- crossing
    from the energy-diffusion-limited to the spatial-diffusion-limited
    regime, i.e. through the Kramers turnover, where the simple high-
    friction scaling above is not strictly valid). It also ignores the
    tilt term c: treating delta_V as the c=0 leading-order term is an
    approximation, not an identity, valid when |c| is small relative to
    b (true for most but not necessarily all observed parameter values).

    Because ROC-AUC and PR-AUC are rank-based, what matters for testing
    whether this proxy is useful is whether it recovers the right
    monotonic relationship between (delta_V, gamma, T) and transition
    likelihood -- not whether it reproduces an absolute rate in 1/day
    units, which it is not calibrated to do.
    """
    delta_v = (b ** 2) / (4.0 * a + 1e-8)
    return np.exp(-delta_v / (temp + 1e-8)) / (gamma + 1e-3)


def fit_beta_calibration(y_calib: np.ndarray, psi_calib: np.ndarray):
    """Beta calibration (Kull, Silva Filho & Flach, 2017): a 3-parameter
    recalibration map, logit(p_cal) = a*log(p) - b*log(1-p) + c, fit
    strictly on the calibration set -- never call this with test-set
    data, same discipline as get_temperature_scaler().

    More flexible than temperature scaling's single scalar T: T_calib can
    only apply a symmetric rescaling around logit(p)=0, so it cannot
    correct a calibration error that is worse on one end of the
    probability range than the other. Beta calibration can, because
    log(p) and log(1-p) are fit with independent coefficients rather than
    tied together through one shared T. Implemented as a 2-feature
    logistic regression on (log(p), log(1-p)) -- the model is linear in
    these two features, so this is exactly equivalent to the textbook
    beta-calibration fit, just expressed through sklearn's solver rather
    than a bespoke one.

    Caveat, stated plainly: this has the same parameter count (3) as the
    Psi+Kramers blend (evaluate_auxiliary_signal), which underperformed
    in practice on this project specifically because the calibration
    window contains only one crisis episode -- not enough independent
    evidence to pin down 3 free parameters reliably. The same risk
    applies here. Compare against T_calib's ECE/Brier/log-loss before
    trusting this over the simpler 1-parameter version.
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    p = np.clip(psi_calib, eps, 1 - eps)
    x = np.column_stack([np.log(p), np.log(1 - p)])
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(x, y_calib)
    return clf


def apply_beta_calibration(clf, psi: np.ndarray) -> np.ndarray:
    """Applies a map fit by fit_beta_calibration() to new (e.g. test-set)
    Psi values."""
    eps = 1e-6
    p = np.clip(psi, eps, 1 - eps)
    x = np.column_stack([np.log(p), np.log(1 - p)])
    return clf.predict_proba(x)[:, 1]


def fit_signal_blend(y_calib: np.ndarray, psi_calib: np.ndarray, kramers_calib: np.ndarray,
                      C: float = 1.0):
    """Fits a 2-input logistic regression blending Psi and the Kramers-rate
    proxy, strictly on the calibration set -- never call this with
    test-set data. Same discipline as get_temperature_scaler(): the blend
    weights must never be chosen by looking at the data being evaluated.

    Uses logit(Psi) and log(kramers) as the two linear predictors:
    log-space for the Kramers proxy because it's an exponential-based
    quantity spanning a wide multiplicative range, which a linear model
    handles better in log-space. The regression finds its own sign for
    each input, so the empirically-observed inversion of the raw Kramers
    proxy (see evaluate_auxiliary_signal) does not need to be hand-
    corrected here -- if the calibration set shows the same inversion,
    the fitted coefficient will simply come out negative.

    C controls regularization strength (smaller = more regularized) and
    matters a lot in this specific case: the calibration window
    (2017-2019) contains exactly one labeled crisis episode
    (2018-10..12), so an under-regularized fit (large C) has very little
    independent evidence to constrain 2 free parameters and can overfit
    to that single episode's specific Psi/Kramers relationship, which may
    not transfer to a structurally different test-set crisis (COVID,
    2022). Default here is 1.0 (sklearn's own default, chosen
    deliberately rather than left implicit, given how much it matters
    here). If a run still underperforms Psi alone, try C in {0.1, 0.01}
    next -- but also consider that no amount of regularization fixes an
    intrinsically under-determined fit; a negative result here is
    informative on its own, not just a tuning problem to solve.
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    p = np.clip(psi_calib, eps, 1 - eps)
    x_psi = np.log(p / (1 - p))
    x_kramers = np.log(kramers_calib + eps)
    x = np.column_stack([x_psi, x_kramers])
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(x, y_calib)
    return clf


def apply_signal_blend(clf, psi: np.ndarray, kramers: np.ndarray) -> np.ndarray:
    """Applies a blend fitted by fit_signal_blend() to new psi/kramers
    values (e.g. test-set), returning blended probabilities. Uses the
    exact same feature transform fit_signal_blend used, so what the model
    sees at inference time matches what it was fit on."""
    eps = 1e-6
    p = np.clip(psi, eps, 1 - eps)
    x_psi = np.log(p / (1 - p))
    x_kramers = np.log(kramers + eps)
    x = np.column_stack([x_psi, x_kramers])
    return clf.predict_proba(x)[:, 1]


def evaluate_auxiliary_signal(y_true: np.ndarray, auxiliary_scores: np.ndarray,
                               primary_scores: np.ndarray) -> dict:
    """Diagnostic for a candidate auxiliary signal (e.g. the Kramers-rate
    proxy) against the primary signal (Psi) already in use.

    Reports the auxiliary signal's own standalone ROC-AUC/PR-AUC (is it
    informative at all on its own?) and its correlation with the primary
    signal (is it redundant with Psi, or does it carry independent
    information?). A signal with decent standalone AUC AND low
    correlation with Psi is the interesting case -- it means combining
    the two (e.g. via a small logistic regression fit on the calibration
    set, analogous to how T_calib is fit) is worth trying as a follow-up
    experiment. A signal that's highly correlated with Psi is not
    necessarily useless, but is unlikely to move the needle if blended.

    Direction handling: unlike a signal designed to point the same way as
    Psi, a physics-derived quantity like a transition rate has no reason
    to have its sign pre-aligned with "higher = more crisis-like". ROC-AUC
    is symmetric under sign flip (an AUC of 0.15 is exactly as informative
    as 0.85, just inverted -- flip the decision rule, not the data), so
    this checks both directions and reports whichever is more informative,
    together with which direction that was. PR-AUC is NOT symmetric under
    sign flip the same way (it depends on which class is "positive"), so
    it is recomputed on the sign-corrected scores rather than derived
    algebraically from the uncorrected value.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    raw_auc = float(roc_auc_score(y_true, auxiliary_scores))
    flipped = raw_auc < 0.5
    effective_scores = -auxiliary_scores if flipped else auxiliary_scores
    effective_auc = float(roc_auc_score(y_true, effective_scores))
    effective_pr_auc = float(average_precision_score(y_true, effective_scores))

    return {
        "raw_roc_auc": raw_auc,
        "direction_flipped": flipped,
        "auxiliary_roc_auc": effective_auc,
        "auxiliary_pr_auc": effective_pr_auc,
        "correlation_with_primary": float(np.corrcoef(auxiliary_scores, primary_scores)[0, 1]),
    }

