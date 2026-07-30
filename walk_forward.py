"""Walk-forward (expanding-window) validation for the PhaseIndex detector.

Motivation, stated plainly: the standard pipeline's single test window
(2020->end) contains only 2 largely independent crisis episodes, which
puts a hard floor on how precisely ANY variant can be compared -- the
block-bootstrap CI on PR-AUC spans roughly [0.43, 0.93]. This harness
retrains the model on expanding historical windows and evaluates each
fold strictly out-of-sample, raising the number of out-of-sample crisis
episodes from 2 to ~8. It changes NOTHING about the model itself: same
training function, same features, same labels. It is an instrument, not
an intervention.

Usage (Colab or local):
    python walk_forward.py

Configuration is read from main_v304_soft_labels.CFG, including all
experimental flags -- the active flags are printed in a banner at start,
so a run's identity is always visible in its log (no more silently
stale flags). To compare a variant against baseline, run this twice:
once with all experimental flags False, once with the variant's flag
True, and compare fold by fold.

Per-fold protocol:
  1. Train a CFG.ENSEMBLE_SEEDS ensemble on all data up to train_end.
  2. Fit a temperature scalar on the LAST `CALIB_TAIL_DAYS` trading days
     of the training window (in-sample calibration, identical protocol
     across folds and variants -- disclosed, and fine for cross-fold
     comparison since every fold and variant is treated the same way).
  3. Run stateful inference on [test_start, test_end], strictly after
     train_end.
  4. Report threshold-free metrics (ROC-AUC, PR-AUC, Cohen's d) and
     thresholded metrics at the analytic tau* on the calibrated
     probabilities, plus per-crisis recall for every labeled crisis
     inside the fold's test window.

The proxy-calibration ensemble from the main pipeline is NOT trained
here (it doubles compute and adds nothing to cross-fold comparison).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

import main_v304_soft_labels as mvsl

# ---------------------------------------------------------------------------
# Fold definitions: (train_end, test_start, test_end), all inclusive.
# Expanding window: each fold trains on everything from the start of the
# sample to train_end. Test windows are chosen to (a) never overlap any
# training data, (b) jointly cover 8 labeled crisis episodes:
#   fold 1: Subprime onset 2007 + GFC 2008-09
#   fold 2: US downgrade / EU debt 2011
#   fold 3: August 2015 selloff
#   fold 4: Q4 2018 selloff
#   fold 5: COVID-19 crash 2020
#   fold 6: 2022 rate-hike drawdown
# 2010's flash-crash window sits inside fold 1's training gap -> fold 2's
# training data, never in a test window; acceptable (it is the least
# distinct episode of the list).
# ---------------------------------------------------------------------------
FOLDS = [
    ("2006-12-31", "2007-01-01", "2009-12-31"),
    ("2010-12-31", "2011-01-01", "2012-12-31"),
    ("2014-12-31", "2015-01-01", "2016-12-31"),
    ("2017-12-31", "2018-01-01", "2019-12-31"),
    ("2019-12-31", "2020-01-01", "2021-12-31"),
    ("2021-12-31", "2022-01-01", "2023-12-31"),
]

CALIB_TAIL_DAYS = 504  # ~2 trading years at the end of each training window

EXPERIMENTAL_FLAGS = [
    "STRICT_CALIBRATION_HOLDOUT", "PROXY_CALIBRATION", "BETA_CALIBRATION",
    "INCLUDE_SPECTRAL_FEATURES", "ASYMMETRIC_SOFT_LABELS",
    "KINETIC_ENERGY_PENALTY", "MACRO_SKIP_CONNECTION", "FDT2_COLORED_NOISE",
    "TCN_VELOCITY_ENCODER", "DYNAMIC_K_FN",
]


def print_flag_banner() -> dict:
    """Prints every experimental flag's current value, loudly. A run's
    log should never leave you guessing which variant it was."""
    flags = {}
    print("=" * 70)
    print("ACTIVE CONFIGURATION (walk-forward run identity)")
    print("-" * 70)
    for name in EXPERIMENTAL_FLAGS:
        value = getattr(mvsl.CFG, name, None)
        flags[name] = bool(value)
        marker = "  <-- ACTIVE" if value else ""
        print(f"  {name:28s} = {value}{marker}")
    print(f"  {'ENSEMBLE_SEEDS':28s} = {mvsl.CFG.ENSEMBLE_SEEDS}")
    print(f"  {'SOFT_LABEL_SIGMA':28s} = {mvsl.CFG.SOFT_LABEL_SIGMA}")
    print("=" * 70)
    return flags


def crises_in_window(index: pd.DatetimeIndex) -> list:
    """Labeled crises overlapping the given index, with display names."""
    out = []
    for start, end in mvsl.CRISES:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if s <= index.max() and e >= index.min():
            name = mvsl.CRISIS_NAMES.get((start, end), f"{start}..{end}")
            out.append({"window": (start, end), "name": name})
    return out


def evaluate_fold(fold_id, train_end, test_start, test_end,
                  dates, X, y_hard, y_soft, persistence=None) -> dict:
    train_mask = dates <= pd.Timestamp(train_end)
    test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    if n_train < 1000 or n_test < 100:
        raise ValueError(f"fold {fold_id}: window too small (train={n_train}, test={n_test})")

    X_train, y_train_soft = X[train_mask], y_soft[train_mask]
    X_test, y_test_hard = X[test_mask], y_hard[test_mask]
    persistence_train = persistence[train_mask] if persistence is not None else None
    test_index = dates[test_mask]
    n_crisis = int(y_test_hard.sum())
    n_train_crisis = int(y_hard[train_mask].sum())

    print(f"\n--- Fold {fold_id}: train ..{train_end} ({n_train}d, {n_train_crisis} crisis days) | "
          f"test {test_start}..{test_end} ({n_test}d, {n_crisis} crisis days) ---")
    if n_train_crisis == 0:
        print("    WARNING: ZERO crisis days in this fold's TRAINING window -- the model "
              "cannot learn the positive class here; treat this fold's numbers as void.")

    t0 = time.time()
    models = []
    for seed in mvsl.CFG.ENSEMBLE_SEEDS:
        model, _ = mvsl.train_one(seed, X_train, y_train_soft, mvsl.CFG.DEVICE,
                                   persistence_tr=persistence_train)
        models.append(model)
    print(f"    ensemble trained in {time.time() - t0:.0f}s")

    # In-sample tail calibration, identical protocol for every fold/variant.
    tail = slice(max(0, n_train - CALIB_TAIL_DAYS), n_train)
    psi_tail = mvsl.run_stateful_inference(models, X_train[tail], mvsl.CFG.DEVICE, mvsl.CFG.LATENT_DIM)
    y_tail_hard = y_hard[train_mask][tail]
    if y_tail_hard.sum() > 0:
        t_calib = mvsl.get_temperature_scaler(psi_tail, y_tail_hard)
    else:
        t_calib = 1.0  # no crisis in the tail: identity temperature, disclosed
    psi_test = mvsl.run_stateful_inference(models, X_test, mvsl.CFG.DEVICE, mvsl.CFG.LATENT_DIM)
    psi_clipped = np.clip(psi_test, 1e-6, 1 - 1e-6)
    p_calib = expit(logit(psi_clipped) / t_calib)

    tau = 1.0 / (1.0 + mvsl.CFG.ASYM_CRISIS_UNDER_K)
    result = {
        "fold": fold_id, "train_end": train_end,
        "test_start": test_start, "test_end": test_end,
        "n_train": n_train, "n_test": n_test, "n_crisis_days": n_crisis,
        "n_train_crisis_days": n_train_crisis,
        "t_calib": float(t_calib),
        "crises": [],
    }

    if n_crisis == 0 or n_crisis == n_test:
        # Degenerate window: rank metrics undefined. Report and move on.
        result.update({"roc_auc": None, "pr_auc": None, "cohens_d": None,
                        "recall": None, "precision": None, "f1": None})
        print("    WARNING: single-class test window; rank metrics undefined for this fold.")
        return result

    pred = (p_calib >= tau).astype(int)
    tp = int(((pred == 1) & (y_test_hard == 1)).sum())
    fp = int(((pred == 1) & (y_test_hard == 0)).sum())
    fn = int(((pred == 0) & (y_test_hard == 1)).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    mu1, mu0 = psi_test[y_test_hard == 1].mean(), psi_test[y_test_hard == 0].mean()
    pooled = np.sqrt((psi_test[y_test_hard == 1].var() + psi_test[y_test_hard == 0].var()) / 2)
    cohens_d = float((mu1 - mu0) / max(pooled, 1e-12))

    result.update({
        "roc_auc": float(roc_auc_score(y_test_hard, psi_test)),
        "pr_auc": float(average_precision_score(y_test_hard, psi_test)),
        "cohens_d": cohens_d,
        "recall": float(recall), "precision": float(precision), "f1": float(f1),
    })

    for crisis in crises_in_window(test_index):
        s, e = pd.Timestamp(crisis["window"][0]), pd.Timestamp(crisis["window"][1])
        cmask = (test_index >= s) & (test_index <= e) & (y_test_hard == 1)
        if cmask.sum() > 0:
            c_recall = float((p_calib[cmask] >= tau).mean())
            result["crises"].append({"name": crisis["name"], "n_days": int(cmask.sum()),
                                      "recall_calibrated": c_recall})

    print(f"    ROC-AUC={result['roc_auc']:.4f}  PR-AUC={result['pr_auc']:.4f}  "
          f"d={result['cohens_d']:.2f}  recall={recall:.3f}  precision={precision:.3f}")
    for c in result["crises"]:
        print(f"      {c['name']:32s} recall={c['recall_calibrated']*100:5.1f}%  ({c['n_days']}d)")
    return result


def main() -> None:
    flags = print_flag_banner()

    prices, all_prices = mvsl.load_price_data(mvsl.CFG)
    returns, features = mvsl.build_feature_matrix(prices, all_prices)
    hard_labels = mvsl.make_hard_labels(features.index)
    soft_labels = mvsl.build_soft_labels(features.index)
    common = features.index.intersection(hard_labels.index)
    X = features.loc[common].values.astype(np.float32)
    y_hard = hard_labels.loc[common].values
    y_soft = soft_labels.loc[common].values.astype(np.float32)
    dates = common
    # Computed unconditionally (cheap, no model involved) so every fold
    # has it available; evaluate_fold only actually uses it when
    # CFG.DYNAMIC_K_FN is True (same "always compute, flag decides
    # whether it's used" pattern as everywhere else in this file).
    persistence = mvsl.compute_credit_persistence(all_prices).reindex(common).fillna(0.0).values

    out_path = Path("results")
    out_path.mkdir(exist_ok=True)
    fname = out_path / f"walk_forward_{'_'.join(k for k, v in flags.items() if v) or 'baseline'}.json"
    print(f"\nWriting to {fname} after EVERY fold (not just at the end) -- if this "
          f"run is interrupted (Colab disconnect, etc.), whatever folds finished "
          f"before the interruption are already saved there. Re-open it directly "
          f"rather than relying on scrollback.\n")

    results = []
    t_start = time.time()
    for i, (train_end, test_start, test_end) in enumerate(FOLDS, 1):
        results.append(evaluate_fold(i, train_end, test_start, test_end,
                                      dates, X, y_hard, y_soft, persistence=persistence))
        # Incremental save: overwrite the file after every fold, not once at
        # the end. A run that dies partway through (disconnect, OOM, etc.)
        # still leaves every fold that DID finish safely on disk.
        partial = {"flags": flags, "calib_tail_days": CALIB_TAIL_DAYS,
                   "ensemble_seeds": list(mvsl.CFG.ENSEMBLE_SEEDS),
                   "folds_completed": len(results), "folds_total": len(FOLDS),
                   "folds": results}
        with open(fname, "w") as f:
            json.dump(partial, f, indent=2)

    valid = [r for r in results if r["roc_auc"] is not None]
    print("\n" + "=" * 70)
    print(f"WALK-FORWARD SUMMARY ({len(valid)}/{len(results)} evaluable folds, "
          f"{time.time() - t_start:.0f}s total)")
    print("-" * 70)
    print(f"{'fold':>4} {'test window':>22} {'ROC-AUC':>8} {'PR-AUC':>8} "
          f"{'d':>6} {'recall':>7} {'prec':>6}")
    for r in results:
        if r["roc_auc"] is None:
            print(f"{r['fold']:>4} {r['test_start']}..{r['test_end']:>10}   (single-class window)")
            continue
        print(f"{r['fold']:>4} {r['test_start']}..{r['test_end']:>10} "
              f"{r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {r['cohens_d']:>6.2f} "
              f"{r['recall']:>7.3f} {r['precision']:>6.3f}")
    if valid:
        for key in ("roc_auc", "pr_auc", "cohens_d", "recall", "precision"):
            vals = np.array([r[key] for r in valid])
            print(f"  {key:10s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")
    print("=" * 70)
    print("Interpretation notes, printed so they travel with every log:")
    print("  - Folds differ in training size AND crisis type; the fold-to-fold")
    print("    spread mixes both. Compare VARIANTS fold-by-fold (paired), not")
    print("    a variant's mean against another window's mean.")
    print("  - Calibration is in-sample-tail by construction here, identical")
    print("    across folds/variants; absolute calibrated recall is optimistic,")
    print("    its BETWEEN-VARIANT differences are the meaningful quantity.")
    print(f"\nSaved: {fname}")


if __name__ == "__main__":
    main()
