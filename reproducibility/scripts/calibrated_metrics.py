"""
calibrated_metrics.py

F1 and ECE for the baselines in baseline_models.py, computed with the
SAME protocol walk_forward.py's evaluate_fold() uses for PhaseIndex
itself -- not a separately-invented one. Verified line-by-line against
evaluate_fold in walk_forward.py before writing this:

  1. Calibration fit ONLY on an in-sample tail of the TRAINING window
     (CALIB_TAIL_DAYS = 504 trading days, ~2 years, ending at train_end)
     -- never on the test window. Same causal, no-leakage logic as
     everything else in this codebase.

  2. PhaseIndex calibrates via single-parameter TEMPERATURE scaling
     (expit(logit(psi)/T)), which needs a [0,1], logit-able input.
     VIX level, GARCH conditional vol, and the trend score are not
     probabilities and are not bounded to [0,1] -- logit() is undefined
     on them. So this module uses 1D Platt scaling (logistic regression
     on the raw score) instead, which is the standard way to turn an
     arbitrary real-valued decision score into a probability, and
     reduces to the same idea for a score that IS already a probability
     (Markov-switching's filtered P(crisis regime)). Different
     calibration technique, identical role in the pipeline and identical
     evaluation protocol once a probability comes out the other end.

  3. Threshold at tau = 1 / (1 + CFG.ASYM_CRISIS_UNDER_K) -- the fixed
     operating point PhaseIndex itself is scored at (CFG.ASYM_CRISIS_
     UNDER_K = 3.0 -> tau = 0.25). Deliberately NOT an F1-maximizing
     threshold search per baseline: PhaseIndex doesn't get to search for
     its best threshold either, so giving baselines that freedom would
     stop being a fair comparison.

  4. ECE via evaluation_metrics.expected_calibration_error (imported
     directly, not reimplemented), n_bins=10 -- same function, same
     number of bins as everywhere else this codebase reports ECE.

Usage: get a baseline's score over the calibration tail AND over the
test window by calling its function from baseline_models.py TWICE --
no changes needed to baseline_models.py itself:

    from baseline_models import garch_signal
    tail_score = garch_signal(all_prices, train_end, calib_tail_start, train_end)
    test_score = garch_signal(all_prices, train_end, test_start, test_end)

(passing test_end=train_end makes the function fit on the same training
data and filter forward only up to train_end, returning the score over
just the tail slice you ask for -- the same causal fit-then-filter
logic already verified in baseline_models.py, just called with a
different right-hand boundary.)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from main_v304_soft_labels import CFG
from evaluation_metrics import expected_calibration_error


def calibrate_and_score(score_calib_tail: np.ndarray, y_calib_tail_hard: np.ndarray,
                         score_test: np.ndarray, y_test_hard: np.ndarray,
                         asym_crisis_under_k: float = None) -> dict | None:
    """Returns {f1, precision, recall, ece, tau, n_calib} or None if the
    calibration tail has zero crisis days (can't fit a 2-class Platt
    model -- same degenerate case walk_forward.py's own temperature
    scaling guards against with `if y_tail_hard.sum() > 0`)."""
    y_calib_tail_hard = np.asarray(y_calib_tail_hard)
    y_test_hard = np.asarray(y_test_hard)

    if y_calib_tail_hard.sum() == 0:
        return None

    k = asym_crisis_under_k if asym_crisis_under_k is not None else CFG.ASYM_CRISIS_UNDER_K
    tau = 1.0 / (1.0 + k)

    clf = LogisticRegression(C=1e10, max_iter=2000)
    clf.fit(np.asarray(score_calib_tail).reshape(-1, 1), y_calib_tail_hard)
    p_test = clf.predict_proba(np.asarray(score_test).reshape(-1, 1))[:, 1]

    pred = (p_test >= tau).astype(int)
    tp = int(((pred == 1) & (y_test_hard == 1)).sum())
    fp = int(((pred == 1) & (y_test_hard == 0)).sum())
    fn = int(((pred == 0) & (y_test_hard == 1)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    ece = expected_calibration_error(y_test_hard, p_test, n_bins=10)

    return {"f1": float(f1), "precision": float(precision), "recall": float(recall),
            "ece": float(ece), "tau": float(tau), "n_calib": int(len(y_calib_tail_hard))}
