"""
run_baseline_comparison.py

Runs every baseline in baseline_models.py through PhaseIndex's own 6
walk-forward folds -- imported directly from walk_forward.py, not
redefined, so the two can never silently drift apart -- and reports
ROC-AUC / PR-AUC / F1 / ECE per fold, directly comparable to the numbers
walk_forward.py itself writes to results/walk_forward_*.json.

F1 and ECE need a calibrated probability and a decision threshold, which
ROC-AUC/PR-AUC do not -- see calibrated_metrics.py for exactly how that
mirrors evaluate_fold()'s own protocol (in-sample tail calibration, the
same fixed tau PhaseIndex is scored at). Each baseline function from
baseline_models.py is called TWICE per fold: once over the calibration
tail (CALIB_TAIL_DAYS trading days ending at train_end), once over the
real test window -- baseline_models.py itself is unchanged.

Labels are built from all_prices.index rather than the engineered
feature matrix, since these baselines don't need PhaseIndex's features.
Sample sizes per fold will differ very slightly from PhaseIndex's own
(each baseline has its own warm-up: GARCH/Markov need enough return
history to fit, trend_ma200 needs `window` days, VIX needs none) --
expected and harmless, not a leak.

Place this file next to main_v304_soft_labels.py, walk_forward.py,
evaluation_metrics.py, baseline_models.py, and calibrated_metrics.py,
then run (same environment as walk_forward.py -- Colab or local):
    python run_baseline_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import main_v304_soft_labels as mvsl
from walk_forward import FOLDS, CALIB_TAIL_DAYS
from baseline_models import BASELINES
from calibrated_metrics import calibrate_and_score


def _calib_tail_start(all_prices_index: pd.DatetimeIndex, train_end: str) -> str:
    """Last CALIB_TAIL_DAYS trading days ending at train_end, by row
    position in the real trading calendar -- not a calendar-day offset --
    same definition walk_forward.py's own `tail = slice(n_train -
    CALIB_TAIL_DAYS, n_train)` uses."""
    train_end_pos = all_prices_index.searchsorted(pd.Timestamp(train_end), side="right") - 1
    tail_start_pos = max(0, train_end_pos - CALIB_TAIL_DAYS + 1)
    return all_prices_index[tail_start_pos].strftime("%Y-%m-%d")


def main() -> None:
    print("Loading price data (same call walk_forward.py itself uses)...")
    _, all_prices = mvsl.load_price_data(mvsl.CFG)
    hard_labels_full = mvsl.make_hard_labels(all_prices.index)

    results = {name: [] for name in BASELINES}

    for fold_id, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        y_test = hard_labels_full.loc[test_start:test_end]
        if y_test.nunique() < 2:
            print(f"\nfold {fold_id}: single-class test window, skipping")
            continue

        tail_start = _calib_tail_start(all_prices.index, train_end)
        y_tail = hard_labels_full.loc[tail_start:train_end]

        print(f"\n--- fold {fold_id}: test {test_start}..{test_end} "
              f"(calib tail {tail_start}..{train_end}) ---")
        for name, fn in BASELINES.items():
            try:
                score_test = fn(all_prices, train_end, test_start, test_end)
                aligned = pd.DataFrame({"y": y_test, "score": score_test}).dropna()
                auc = roc_auc_score(aligned["y"], aligned["score"])
                pr_auc = average_precision_score(aligned["y"], aligned["score"])

                row = {"fold": fold_id, "roc_auc": auc, "pr_auc": pr_auc,
                       "n_test": int(len(aligned))}

                score_tail = fn(all_prices, train_end, tail_start, train_end)
                aligned_tail = pd.DataFrame({"y": y_tail, "score": score_tail}).dropna()
                calib = calibrate_and_score(aligned_tail["score"].values, aligned_tail["y"].values,
                                             aligned["score"].values, aligned["y"].values)
                if calib is None:
                    print(f"    {name:18s} ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
                          f"F1=n/a (calib tail has 0 crisis days)")
                    row.update({"f1": None, "ece": None})
                else:
                    row.update(calib)
                    print(f"    {name:18s} ROC-AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  "
                          f"F1={calib['f1']:.4f}  ECE={calib['ece']:.4f}  (tau={calib['tau']:.3f})")
                results[name].append(row)
            except Exception as e:
                print(f"    {name:18s} FAILED: {e}")
                results[name].append({"fold": fold_id, "roc_auc": None, "pr_auc": None,
                                       "f1": None, "ece": None, "error": str(e)})

    print("\n" + "=" * 70)
    print("SUMMARY (mean +/- std across evaluable folds)")
    print("-" * 70)
    for name, folds in results.items():
        for metric in ("roc_auc", "pr_auc", "f1", "ece"):
            vals = [f[metric] for f in folds if f.get(metric) is not None]
            if vals:
                print(f"  {name:18s} {metric.upper():8s} = {np.mean(vals):.4f} +/- {np.std(vals):.4f}  "
                      f"(n={len(vals)} folds)")
        print()
    print("  Compare against PhaseIndex's own walk-forward numbers in your")
    print("  own results/walk_forward_*.json (ROC-AUC 0.883 +/- 0.062 as of")
    print("  the last recorded run; walk_forward.py's evaluate_fold() already")
    print("  reports f1 per fold the same way -- it does not currently report")
    print("  ECE per fold, only in the separate single-window pipeline).")
    print("=" * 70)

    out_path = Path("results") / "baseline_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
