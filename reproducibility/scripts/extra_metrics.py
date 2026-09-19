"""
extra_metrics.py

Two additions that operate on scores already produced, so both cost
essentially nothing on top of a run that was happening anyway.

1. TRANSITION-AUC. Detection restricted to the onset of each crisis
   rather than the whole episode. Full-window ROC-AUC rewards a
   detector for correctly labeling the middle of a two-year bear market,
   which is the easy part; a detector that only notices a crisis three
   weeks in can still score well. Transition-AUC asks the harder
   question: over the `pre` trading days before onset plus the first
   `post` days of the episode, can the score separate the two?

   This metric must be reported whatever it says. It was proposed on the
   expectation that it would favour the model over a moving-average
   rule, and a metric adopted because its result is anticipated is worth
   nothing. Note also that the full-window lead-time measurements on the
   six folds give PhaseIndex no early crossing at all on two of seven
   crises, so the expectation is not supported in advance by the
   evidence already in hand.

2. RANK ENSEMBLE. A logistic blend of Psi and the Kramers rate R_K was
   tried previously and failed to beat Psi alone: the calibration window
   holds one crisis episode, too few events to fit two weights. A rank
   average has no weights to fit at all, so it sidesteps exactly that
   failure.

   One precaution that matters here. Psi and R_K are reported as
   correlating at -0.15 to -0.22 while both are positively predictive,
   which cannot be taken at face value as an orientation. Averaging two
   ranks that point opposite ways cancels signal instead of combining
   it, so `orient` below fixes each component's sign against the
   training labels before averaging, rather than assuming.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def transition_auc(score: np.ndarray, dates: pd.DatetimeIndex, y_hard: np.ndarray,
                    crisis_starts, pre: int = 60, post: int = 20) -> dict:
    """ROC-AUC restricted to onset windows. For each crisis start, takes
    the `pre` trading days immediately before it that are not themselves
    labeled crisis days (negatives) and the first `post` labeled days of
    the episode (positives), pools them across crises, and scores once.

    Returns per-crisis AUCs and the pooled AUC. Pooled is the number to
    quote: per-crisis windows hold at most pre+post points, which is too
    few to read individually.
    """
    dates = pd.DatetimeIndex(dates)
    pos_idx, neg_idx, per_crisis = [], [], {}

    for start in crisis_starts:
        s = pd.Timestamp(start)
        after = np.where((dates >= s) & (y_hard == 1))[0]
        before = np.where((dates < s) & (y_hard == 0))[0]
        if len(after) == 0 or len(before) == 0:
            continue
        p = after[:post]
        n = before[-pre:]
        pos_idx.extend(p.tolist())
        neg_idx.extend(n.tolist())
        if len(p) > 0 and len(n) > 0:
            yy = np.r_[np.ones(len(p)), np.zeros(len(n))]
            ss = np.r_[score[p], score[n]]
            per_crisis[str(start)] = float(roc_auc_score(yy, ss))

    if not pos_idx or not neg_idx:
        return {"pooled": None, "per_crisis": per_crisis, "n_pos": 0, "n_neg": 0}

    y = np.r_[np.ones(len(pos_idx)), np.zeros(len(neg_idx))]
    s = np.r_[score[pos_idx], score[neg_idx]]
    return {"pooled": float(roc_auc_score(y, s)), "per_crisis": per_crisis,
            "n_pos": len(pos_idx), "n_neg": len(neg_idx)}


def orient(score_train: np.ndarray, y_train: np.ndarray) -> float:
    """+1 or -1, whichever makes the score positively predictive ON THE
    TRAINING LABELS. Fitting the sign on training data keeps the test
    evaluation honest; a sign chosen on test data would be a one-bit
    leak, small but real."""
    return 1.0 if roc_auc_score(y_train, score_train) >= 0.5 else -1.0


def rank_ensemble(psi_test: np.ndarray, rk_test: np.ndarray,
                   psi_train: np.ndarray, rk_train: np.ndarray,
                   y_train: np.ndarray, w: float = 0.5) -> np.ndarray:
    """Weight-free combination: average of within-window normalized
    ranks, each component oriented on training labels first. Ranks are
    divided by n so the two components are on a common [0, 1] scale
    regardless of window length."""
    s_psi = orient(psi_train, y_train)
    s_rk = orient(rk_train, y_train)
    n = len(psi_test)
    r_psi = rankdata(s_psi * psi_test) / n
    r_rk = rankdata(s_rk * rk_test) / n
    return w * r_psi + (1.0 - w) * r_rk
