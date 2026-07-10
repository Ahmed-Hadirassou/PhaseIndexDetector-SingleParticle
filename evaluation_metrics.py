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
