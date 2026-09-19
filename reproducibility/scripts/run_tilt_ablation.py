"""
run_tilt_ablation.py

Runs the four branches of tilt_ablation.py through the same six
walk-forward folds, same labels, same causal boundaries, same
calibration protocol as walk_forward.py.

WHAT THIS RECORDS THAT EARLIER RUNS DID NOT

Every previous result in this project reports the five-seed ensemble
only, which means seed variance has never been separated from
fold variance. That gap is worth closing here because it is nearly free:
the seeds are being trained anyway, so storing each seed's individual
test score alongside the ensemble's costs one extra inference pass per
seed and nothing else. It matters for reading this experiment in
particular. On fold 5 the single-seed and five-seed numbers in earlier
work differ by roughly 0.044 ROC-AUC, which is larger than any effect
this ablation is likely to detect, so a branch difference smaller than
the seed spread should not be called a difference at all.

RUNTIME

4 branches x 6 folds x N_SEEDS trainings. At roughly 30 to 40 seconds
per training on the hardware used for the earlier audit, the default
N_SEEDS = 5 lands near 1.5 to 2 hours. Results are written after every
branch, so an interrupted run keeps whatever finished.

Cutting to N_SEEDS = 1 finishes in about 20 minutes but, given the seed
spread noted above, will not support a conclusion. Run it that way only
to confirm the pipeline works end to end.

Place next to main_v304_soft_labels.py, walk_forward.py,
tilt_ablation.py and extra_metrics.py, then:
    python run_tilt_ablation.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score

import main_v304_soft_labels as mvsl
import evaluation_metrics as em
from main_v304_soft_labels import CFG
from walk_forward import FOLDS, CALIB_TAIL_DAYS
import tilt_ablation as tilt
from extra_metrics import transition_auc

N_SEEDS = 5
SEEDS = (42, 123, 456, 789, 1011)[:N_SEEDS]
BRANCHES = ("v2", "horizon", "physics", "coupled")
OUT = Path("results") / "tilt_ablation.json"


def evaluate_branch_fold(mode, fold_id, X, dates, y_hard, y_soft, crisis_starts):
    train_end, test_start, test_end = FOLDS[fold_id - 1]
    train_mask = dates <= pd.Timestamp(train_end)
    test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr_hard, y_te_hard = y_hard[train_mask], y_hard[test_mask]
    n_train = int(train_mask.sum())

    if y_te_hard.sum() in (0, len(y_te_hard)):
        return {"fold": fold_id, "roc_auc": None, "note": "single-class test window"}

    models, per_seed = [], []
    for s in SEEDS:
        m, _ = mvsl.train_one(s, X_tr, y_soft[train_mask], CFG.DEVICE)
        models.append(m)
        psi_s = mvsl.run_stateful_inference([m], X_te, CFG.DEVICE, CFG.LATENT_DIM)
        per_seed.append(float(roc_auc_score(y_te_hard, psi_s)))

    psi_te = mvsl.run_stateful_inference(models, X_te, CFG.DEVICE, CFG.LATENT_DIM)

    # Same calibration protocol as walk_forward.evaluate_fold: temperature
    # fit on the in-sample tail, identity fallback when that tail holds no
    # crisis day. Kept identical so these numbers sit beside the paper's.
    tail = slice(max(0, n_train - CALIB_TAIL_DAYS), n_train)
    y_tail = y_tr_hard[tail]
    t_calib = 1.0
    if y_tail.sum() > 0:
        psi_tail = mvsl.run_stateful_inference(models, X_tr[tail], CFG.DEVICE, CFG.LATENT_DIM)
        t_calib = mvsl.get_temperature_scaler(psi_tail, y_tail)
    p_cal = expit(logit(np.clip(psi_te, 1e-6, 1 - 1e-6)) / t_calib)

    tau = 1.0 / (1.0 + CFG.ASYM_CRISIS_UNDER_K)
    pred = (p_cal >= tau).astype(int)
    tp = int(((pred == 1) & (y_te_hard == 1)).sum())
    fp = int(((pred == 1) & (y_te_hard == 0)).sum())
    fn = int(((pred == 0) & (y_te_hard == 1)).sum())
    rec = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)

    ta = transition_auc(psi_te, dates[test_mask], y_te_hard, crisis_starts)

    return {
        "fold": fold_id,
        "roc_auc": float(roc_auc_score(y_te_hard, psi_te)),
        "pr_auc": float(average_precision_score(y_te_hard, psi_te)),
        "f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "ece": float(em.expected_calibration_error(y_te_hard, p_cal, n_bins=10)),
        "transition_auc": ta["pooled"],
        "per_seed_roc_auc": per_seed,
        "seed_std": float(np.std(per_seed)),
        "t_calib": float(t_calib),
        "n_test": int(test_mask.sum()),
    }


def main() -> None:
    print("Loading price data and building features...")
    prices, all_prices = mvsl.load_price_data(CFG)
    returns, features = mvsl.build_feature_matrix(prices, all_prices)
    hard = mvsl.make_hard_labels(features.index)
    soft = mvsl.build_soft_labels(features.index)
    common = features.index.intersection(hard.index)

    X0 = features.loc[common].values.astype(np.float32)
    y_hard = hard.loc[common].values
    y_soft = soft.loc[common].values.astype(np.float32)
    dates = common

    M = tilt.compute_trend_signal(prices, tilt.TREND_TAU).reindex(common).ffill().fillna(0.0).values
    X, slow_idx, slow_dim = tilt.append_trend_column(X0, M)
    print(f"feature matrix {X0.shape} -> {X.shape}; slow block {CFG.SLOW_DIM} -> {slow_dim}")

    orig_slow_idx, orig_slow_dim = list(CFG.SLOW_IDX), CFG.SLOW_DIM
    CFG.SLOW_IDX, CFG.SLOW_DIM = slow_idx, slow_dim

    crisis_starts = [s for s, _ in mvsl.CRISES]
    results, t0 = {}, time.time()
    try:
        for mode in BRANCHES:
            tilt.install(mode)
            print(f"\n{'=' * 64}\nBRANCH {mode}\n{'=' * 64}")
            rows = []
            for fold_id in range(1, len(FOLDS) + 1):
                r = evaluate_branch_fold(mode, fold_id, X, dates, y_hard, y_soft, crisis_starts)
                rows.append(r)
                if r.get("roc_auc") is None:
                    print(f"  fold {fold_id}: {r.get('note')}")
                else:
                    ta = r["transition_auc"]
                    print(f"  fold {fold_id}: ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}  "
                          f"F1={r['f1']:.4f}  ECE={r['ece']:.4f}  "
                          f"transAUC={ta:.4f}" if ta is not None else
                          f"  fold {fold_id}: ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}  "
                          f"F1={r['f1']:.4f}  ECE={r['ece']:.4f}  transAUC=n/a")
                    print(f"            per-seed ROC {['%.4f' % v for v in r['per_seed_roc_auc']]}  "
                          f"(sd {r['seed_std']:.4f})")
            results[mode] = rows
            OUT.parent.mkdir(exist_ok=True)
            with open(OUT, "w") as f:
                json.dump({"seeds": list(SEEDS), "tau": tilt.TREND_TAU, "branches": results}, f, indent=2)
            print(f"  [saved: {OUT}]")
    finally:
        tilt.restore()
        CFG.SLOW_IDX, CFG.SLOW_DIM = orig_slow_idx, orig_slow_dim

    print(f"\n{'=' * 64}\nSUMMARY ({time.time() - t0:.0f}s)\n{'=' * 64}")
    print(f"{'branch':10s} {'ROC-AUC':>16s} {'PR-AUC':>9s} {'F1':>8s} {'ECE':>8s} {'transAUC':>9s} {'seed sd':>9s}")
    for mode, rows in results.items():
        ok = [r for r in rows if r.get("roc_auc") is not None]
        if not ok:
            continue
        roc = np.array([r["roc_auc"] for r in ok])
        sd = np.mean([r["seed_std"] for r in ok])
        ta = [r["transition_auc"] for r in ok if r["transition_auc"] is not None]
        print(f"{mode:10s} {roc.mean():8.4f}+-{roc.std():.4f} "
              f"{np.mean([r['pr_auc'] for r in ok]):9.4f} "
              f"{np.mean([r['f1'] for r in ok]):8.4f} "
              f"{np.mean([r['ece'] for r in ok]):8.4f} "
              f"{(np.mean(ta) if ta else float('nan')):9.4f} {sd:9.4f}")
    print("\nRead the last column first. A gap between branches smaller than the")
    print("mean within-fold seed spread is not a result, whatever its sign.")
    print("v2 vs horizon  -> was the missing ingredient simply a longer horizon?")
    print("physics vs v2  -> can one scalar replace the network's tilt entirely?")
    print("coupled vs both-> does the hybrid beat each of its own special cases?")


if __name__ == "__main__":
    main()
