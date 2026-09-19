"""
run_overfitting_suite.py

Runs every DISTINCT overfitting diagnostic built across this project,
back to back, and prints one scorecard at the end. "Distinct" is the
operative word: each test below answers a different question, on
purpose, so they cannot all fail or pass together for the same reason --
that's what makes the combination worth more than any one of them.

  Q1. Does the model memorize training data instead of generalizing?
      -> auc_gap, per fold, already computed by walk_forward.py itself
      (not duplicated here -- run that script for this piece, or read
      an existing walk_forward_*.json if you already have one).
  Q2. Same question, but WHEN does it happen during training (dynamic,
      epoch-by-epoch), not just a before/after snapshot?
      -> train_one_with_validation (this suite: SECTION 1)
  Q3. Is there real learnable signal at all, or could random labels
      produce a similar score by chance given only ~13 crisis windows?
      -> permutation_test (SECTION 2)
  Q4. Is performance a fragile spike at exactly the current
      hyperparameters, or does it hold up nearby?
      -> hyperparameter_sensitivity_sweep (SECTION 3)
  Q5. Is the model data-starved -- would more history keep helping, or
      has it already saturated on a small slice?
      -> learning_curve_vs_sample_size (SECTION 4)
  Q6. Does the signal depend narrowly on ONE feature source, or is it
      robust across all six?
      -> feature_group_ablation (SECTION 5)

None of these, alone or together, PROVES the absence of overfitting in
every sense of the word -- said plainly rather than left implicit,
because it matters: no statistical test run on a finished model can
rule out that development decisions (which hyperparameters, which
extensions) were shaped by repeatedly looking at results across this
project's history. That is a property of the RESEARCH PROCESS, not of
the artifact, and nothing below (or anywhere) closes that question by
itself. What this script gives you is five distinct, honest angles on
the artifact -- which is real evidence, not a rubber stamp.

Saves incrementally to results/overfitting_suite.json after EACH
section (same reasoning as walk_forward.py's own incremental save): a
Colab disconnect partway through does not lose whatever finished first.

Expect roughly 45-60 minutes total with the conservative defaults
below, almost all of it SECTION 2 (permutation_test) -- check the
printed per-permutation timing early and Ctrl-C / adjust N_PERMUTATIONS
if your hardware runs slower than that estimate. Every section is
independently commented out-able if you'd rather run this over a few
sittings than one long blocking call.

Place this file next to main_v304_soft_labels.py, walk_forward.py, and
overfitting_diagnostics.py, then run (Colab or local):
    python run_overfitting_suite.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import main_v304_soft_labels as mvsl
from walk_forward import FOLDS, CALIB_TAIL_DAYS
import overfitting_diagnostics as od

# --------------------------------------------------------------------- #
# Conservative defaults -- widen once you've seen real per-test timing. #
# --------------------------------------------------------------------- #
VAL_CURVE_FOLDS = (2, 5)          # one "clean" fold, one COVID-anchored fold
N_PERMUTATIONS = 10               # up from the n=3 exploratory run; still not the
                                   # 20-30 needed for a citable p-value -- this is
                                   # a "everything in one pass" middle ground
PERMUTATION_FOLDS = (1, 2, 3, 4)  # unrelated to the fold-5/6 hyperparameter-
                                   # provenance question -- chosen for compute cost
HP_SWEEP_FOLD = 2
LEARNING_CURVE_FOLD = 2
ABLATION_FOLD = 2

OUT_PATH = Path("results") / "overfitting_suite.json"


def _save(all_results: dict) -> None:
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda o: o.to_dict("records")
                   if isinstance(o, pd.DataFrame) else str(o))
    print(f"  [saved: {OUT_PATH}]")


def main() -> None:
    t_suite_start = time.time()
    results: dict = {}

    print("Loading price data and building features (same call walk_forward.py itself uses)...")
    prices, all_prices = mvsl.load_price_data(mvsl.CFG)
    returns, features = mvsl.build_feature_matrix(prices, all_prices)
    hard_labels = mvsl.make_hard_labels(features.index)
    soft_labels = mvsl.build_soft_labels(features.index)
    common = features.index.intersection(hard_labels.index)
    X = features.loc[common].values.astype(np.float32)
    y_hard = hard_labels.loc[common].values
    y_soft = soft_labels.loc[common].values.astype(np.float32)
    dates = common

    # --- Section 1: train/val loss curves (Q2) --------------------------
    print("\n" + "=" * 70)
    print(f"SECTION 1/5 -- train_one_with_validation on folds {VAL_CURVE_FOLDS}")
    print("=" * 70)
    val_curves = {}
    for fold_id in VAL_CURVE_FOLDS:
        train_end, _, _ = FOLDS[fold_id - 1]
        train_mask = dates <= pd.Timestamp(train_end)
        X_fold, y_fold = X[train_mask], y_soft[train_mask]
        X_tr, y_tr = X_fold[:-CALIB_TAIL_DAYS], y_fold[:-CALIB_TAIL_DAYS]
        X_val, y_val = X_fold[-CALIB_TAIL_DAYS:], y_fold[-CALIB_TAIL_DAYS:]
        print(f"\n  fold {fold_id}:")
        _, train_hist, val_hist = od.train_one_with_validation(
            42, X_tr, y_tr, X_val, y_val, mvsl.CFG.DEVICE, verbose=False)
        best_epoch_idx = int(np.argmin(val_hist))
        val_curves[str(fold_id)] = {"train_loss": train_hist, "val_loss": val_hist,
                                     "val_min_at_checkpoint": best_epoch_idx,
                                     "val_min": val_hist[best_epoch_idx],
                                     "val_final": val_hist[-1],
                                     "rose_after_min": val_hist[-1] > val_hist[best_epoch_idx]}
        print(f"    val_loss minimum at checkpoint {best_epoch_idx + 1}/9 "
              f"({val_hist[best_epoch_idx]:.4f}); final={val_hist[-1]:.4f}")
    results["train_val_curves"] = val_curves
    _save(results)

    # --- Section 2: permutation test (Q3) --------------------------------
    print("\n" + "=" * 70)
    print(f"SECTION 2/5 -- permutation_test, {N_PERMUTATIONS} permutations, "
          f"folds {PERMUTATION_FOLDS}")
    print("=" * 70)
    perm_result = od.permutation_test(X, dates, fold_ids=PERMUTATION_FOLDS,
                                       n_permutations=N_PERMUTATIONS, seeds=(42,))
    results["permutation_test"] = perm_result
    _save(results)

    # --- Section 3: hyperparameter sensitivity (Q4) ----------------------
    print("\n" + "=" * 70)
    print(f"SECTION 3/5 -- hyperparameter_sensitivity_sweep, fold {HP_SWEEP_FOLD}")
    print("=" * 70)
    hp_df = od.hyperparameter_sensitivity_sweep(
        X, dates, latent_dims=(3, 4, 5), verlet_steps=(6,),
        fold_ids=(HP_SWEEP_FOLD,), seeds=(42,))
    results["hyperparameter_sweep"] = hp_df.to_dict("records")
    _save(results)

    # --- Section 4: learning curve vs. sample size (Q5) -------------------
    print("\n" + "=" * 70)
    print(f"SECTION 4/5 -- learning_curve_vs_sample_size, fold {LEARNING_CURVE_FOLD}")
    print("=" * 70)
    lc_df = od.learning_curve_vs_sample_size(
        X, dates, y_hard, y_soft, fold_id=LEARNING_CURVE_FOLD,
        fractions=(0.5, 0.75, 1.0), seeds=(42,))
    results["learning_curve"] = lc_df.to_dict("records")
    _save(results)

    # --- Section 5: feature-group ablation (Q6) ---------------------------
    print("\n" + "=" * 70)
    print(f"SECTION 5/5 -- feature_group_ablation, fold {ABLATION_FOLD}")
    print("=" * 70)
    abl_df = od.feature_group_ablation(prices, all_prices, dates, fold_id=ABLATION_FOLD, seeds=(42,))
    results["feature_ablation"] = abl_df.to_dict("records")
    _save(results)

    # --- Scorecard ----------------------------------------------------
    print("\n" + "=" * 70)
    print(f"SCORECARD ({time.time() - t_suite_start:.0f}s total)")
    print("=" * 70)
    print("Five distinct angles, five distinct answers -- read each on its own")
    print("terms, not as five votes on one verdict:\n")

    for fold_id, c in val_curves.items():
        flag = "ROSE after the minimum" if c["rose_after_min"] else "did not rise after the minimum"
        print(f"  Q2 (fold {fold_id}) train/val curve : val_loss {flag} "
              f"(min={c['val_min']:.4f} @ ckpt {c['val_min_at_checkpoint']+1}/9, "
              f"final={c['val_final']:.4f})")

    p = perm_result["empirical_p_value"]
    print(f"  Q3 permutation test          : real={perm_result['real_mean_auc']:.4f} vs "
          f"null={perm_result['null_mean']:.4f}+/-{perm_result['null_std']:.4f} "
          f"(n={perm_result['n_permutations_completed']} draws, p={p:.3f} -- "
          f"resolution is 1/n, not a precise value at this n)")

    if len(hp_df):
        spread = hp_df["mean_roc_auc"].max() - hp_df["mean_roc_auc"].min()
        print(f"  Q4 hyperparameter sweep      : ROC-AUC range {hp_df['mean_roc_auc'].min():.4f}-"
              f"{hp_df['mean_roc_auc'].max():.4f} across LATENT_DIM (spread={spread:.4f})")

    if len(lc_df):
        trend = "IMPROVED" if lc_df["roc_auc"].iloc[-1] > lc_df["roc_auc"].iloc[0] else "did not improve"
        print(f"  Q5 learning curve             : test ROC-AUC {trend} from "
              f"{lc_df['roc_auc'].iloc[0]:.4f} (n={lc_df['n_train'].iloc[0]}) to "
              f"{lc_df['roc_auc'].iloc[-1]:.4f} (n={lc_df['n_train'].iloc[-1]})")

    if len(abl_df) > 1:
        worst = abl_df.iloc[1:].loc[abl_df.iloc[1:]["drop_from_full"].idxmax()]
        print(f"  Q6 feature ablation           : largest single-group drop = "
              f"{worst['excluded']} ({worst['drop_from_full']:+.4f})")

    print("\n  Q1 (train/test AUC gap, all folds) is NOT in this scorecard --")
    print("  already covered by walk_forward.py's own auc_gap column; read that")
    print("  file's summary instead of duplicating the computation here.")
    print("=" * 70)


if __name__ == "__main__":
    main()
