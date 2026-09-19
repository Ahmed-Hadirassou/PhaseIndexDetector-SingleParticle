"""
run_tilt_physics_resume.py

Physics branch only, all 6 folds, nothing else. Written for exactly one
situation: the physics branch crashed mid-fold-6 on Colab (a disconnect
after 2+ hours, not an exception in the run -- the log simply stops
mid-epoch) after folds 1-5 had already finished and printed.

Two differences from the original run_tilt_ablation.py, both aimed at
not repeating that loss:

1. Saves after EVERY fold, not after the whole branch. The original
   script's coarser save granularity is exactly why folds 1-5 of physics
   were sitting only in scrollback and had to be reconstructed by hand
   after the crash. Coupled has not run yet and faces the identical
   2h+ runtime risk -- this is worth doing here regardless of physics
   specifically.

2. Runs physics only. v2 and horizon are done; re-running them here
   would repeat roughly an hour of finished work for nothing.

evaluate_branch_fold is imported from run_tilt_ablation.py rather than
copied, so fold 6 is computed by the exact same function that produced
folds 1-5 -- no risk of a second, slightly different implementation
quietly drifting from the first one.

Place next to the same files run_tilt_ablation.py needs (the 8 repo
files, tilt_ablation.py, extra_metrics.py, run_tilt_ablation.py) plus
this file, then:
    python run_tilt_physics_resume.py
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
from run_tilt_ablation import evaluate_branch_fold

OUT = Path("results") / "tilt_physics_resume.json"


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

    tilt.install("physics")
    print("\n" + "=" * 64 + "\nBRANCH physics -- resume, all 6 folds\n" + "=" * 64)
    rows, t0 = [], time.time()
    try:
        for fold_id in range(1, len(FOLDS) + 1):
            r = evaluate_branch_fold("physics", fold_id, X, dates, y_hard, y_soft, crisis_starts)
            rows.append(r)
            if r.get("roc_auc") is None:
                print(f"  fold {fold_id}: {r.get('note')}")
            else:
                ta = r["transition_auc"]
                ta_str = f"{ta:.4f}" if ta is not None else "n/a"
                print(f"  fold {fold_id}: ROC={r['roc_auc']:.4f}  PR={r['pr_auc']:.4f}  "
                      f"F1={r['f1']:.4f}  ECE={r['ece']:.4f}  transAUC={ta_str}")
                print(f"            per-seed ROC {['%.4f' % v for v in r['per_seed_roc_auc']]}  "
                      f"(sd {r['seed_std']:.4f})")
            # Saved after EVERY fold -- the one change that matters most here.
            OUT.parent.mkdir(exist_ok=True)
            with open(OUT, "w") as f:
                json.dump({"branch": "physics", "folds_done": [row["fold"] for row in rows],
                           "folds": rows}, f, indent=2)
            print(f"  [saved: {OUT}, {len(rows)}/6 folds]")
    finally:
        tilt.restore()
        CFG.SLOW_IDX, CFG.SLOW_DIM = orig_slow_idx, orig_slow_dim

    ok = [r for r in rows if r.get("roc_auc") is not None]
    if ok:
        roc = np.array([r["roc_auc"] for r in ok])
        print(f"\nphysics: ROC-AUC mean={roc.mean():.4f} std={roc.std():.4f} "
              f"over {len(ok)}/6 folds ({time.time() - t0:.0f}s)")
    print(f"\nMerge {OUT}'s \"folds\" list into your reconstructed JSON's "
          f"physics branch, replacing the incomplete fold 6 entry -- "
          f"folds 1-5 recomputed here should match your reconstructed "
          f"numbers closely (same seeds, same data) but are not required "
          f"to match to the last digit; if any diverge noticeably, trust "
          f"this file, it was saved directly rather than reconstructed "
          f"from scrollback.")


if __name__ == "__main__":
    main()
