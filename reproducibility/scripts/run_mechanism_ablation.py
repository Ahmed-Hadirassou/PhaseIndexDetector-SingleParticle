"""
run_mechanism_ablation.py

Runs the four branches of mechanism_ablation.py (LL / FL / LF / FF)
through the same walk-forward protocol as everything else: 6 folds,
5 seeds, same labels, same calibration, per-seed scores recorded,
transition-AUC included. Saves after EVERY fold from the start.

BRANCHES below can be cut to a subset; e.g. ("LL", "FL") alone answers
the Guttal question (does the well need to move) in half the time, and
LL is the same computation as the earlier "physics" branch, so if that
run is still on disk it can be skipped and its numbers reused.

Runtime: same as the tilt ablation, roughly 55-60 min per branch on
the hardware used so far. Full 2x2 is 3.5-4h; per-fold saves mean a
disconnect costs one fold, not a branch.

Place next to the 8 repo files, tilt_ablation.py, extra_metrics.py,
run_tilt_ablation.py, mechanism_ablation.py and this file:
    python run_mechanism_ablation.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG
from walk_forward import FOLDS
import tilt_ablation as tilt
import mechanism_ablation as mech
from run_tilt_ablation import evaluate_branch_fold, SEEDS

BRANCHES = ("LL", "FL", "LF", "FF")
OUT = Path("results") / "mechanism_ablation.json"


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
    orig_slow_idx, orig_slow_dim = list(CFG.SLOW_IDX), CFG.SLOW_DIM
    CFG.SLOW_IDX, CFG.SLOW_DIM = slow_idx, slow_dim
    crisis_starts = [s for s, _ in mvsl.CRISES]

    results, t0 = {}, time.time()
    try:
        for mode in BRANCHES:
            mech.install(mode)
            print(f"\n{'=' * 64}\nBRANCH {mode}  (potential {'fixed' if mode[0]=='F' else 'per-day'}, "
                  f"thermostat {'fixed' if mode[1]=='F' else 'per-day'})\n{'=' * 64}")
            rows = []
            for fold_id in range(1, len(FOLDS) + 1):
                r = evaluate_branch_fold(mode, fold_id, X, dates, y_hard, y_soft, crisis_starts)
                # one extra cheap training to read the fixed scalars this branch
                # settled on -- interpretation only, not used for any score
                if mode != "LL" and r.get("roc_auc") is not None:
                    train_end = FOLDS[fold_id - 1][0]
                    tr = dates <= pd.Timestamp(train_end)
                    m_probe, _ = mvsl.train_one(SEEDS[0], X[tr], y_soft[tr], CFG.DEVICE)
                    r["learned_scalars_seed0"] = mech.learned_scalars(m_probe)
                rows.append(r)
                if r.get("roc_auc") is None:
                    print(f"  fold {fold_id}: {r.get('note')}")
                else:
                    ta = r["transition_auc"]
                    print(f"  fold {fold_id}: ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}  F1={r['f1']:.4f}  "
                          f"ECE={r['ece']:.4f}  trans={(f'{ta:.4f}' if ta is not None else 'n/a')}  "
                          f"seed_sd={r['seed_std']:.4f}")
                    if "learned_scalars_seed0" in r:
                        print(f"            fixed values: {r['learned_scalars_seed0']}")
                results[mode] = rows
                OUT.parent.mkdir(exist_ok=True)
                with open(OUT, "w") as f:
                    json.dump({"seeds": list(SEEDS), "branches": results}, f, indent=2)
                print(f"  [saved, {len(rows)}/6 folds of {mode}, {time.time() - t0:.0f}s elapsed]")
    finally:
        mech.restore()
        CFG.SLOW_IDX, CFG.SLOW_DIM = orig_slow_idx, orig_slow_dim

    print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
    print(f"{'branch':8s} {'ROC-AUC':>16s} {'PR-AUC':>8s} {'F1':>8s} {'ECE':>8s} {'transAUC':>9s} {'seed sd':>8s}")
    for mode, rows in results.items():
        ok = [r for r in rows if r.get("roc_auc") is not None]
        if not ok:
            continue
        roc = np.array([r["roc_auc"] for r in ok])
        ta = [r["transition_auc"] for r in ok if r["transition_auc"] is not None]
        print(f"{mode:8s} {roc.mean():8.4f}+-{roc.std():.4f} {np.mean([r['pr_auc'] for r in ok]):8.4f} "
              f"{np.mean([r['f1'] for r in ok]):8.4f} {np.mean([r['ece'] for r in ok]):8.4f} "
              f"{(np.mean(ta) if ta else float('nan')):9.4f} {np.mean([r['seed_std'] for r in ok]):8.4f}")
    print("\nFL ~ LL  -> the well never needed to move; crises are the noise rising (Guttal et al. 2016).")
    print("LF ~ LL  -> the thermostat never needed to move; crises are the well flattening.")
    print("FF ~ LL  -> the learned physics is decorative. Serious.")
    print("Neither -> both mechanisms are load-bearing.")
    print("Same rule as always: a gap below the seed sd column is not a gap.")


if __name__ == "__main__":
    main()
