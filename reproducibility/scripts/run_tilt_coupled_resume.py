"""
run_tilt_coupled_resume.py

Coupled branch only, all 6 folds, nothing else. v2, horizon and physics
are already done; running the original run_tilt_ablation.py again would
redo all three (roughly 2.5-3h) before ever reaching coupled.

One difference from the original script, motivated by the physics
branch's own run: that one crashed on Colab mid-fold-6 (a disconnect
after 2+ hours -- the log just stops mid-epoch, not an exception), and
because run_tilt_ablation.py only saves after a whole branch finishes,
folds 1-5 had to be reconstructed by hand from scrollback afterward.
Coupled faces the identical multi-hour runtime, so this script saves
after EVERY fold instead, so a repeat disconnect here loses at most one
fold's work, not the whole branch.

evaluate_branch_fold is imported from run_tilt_ablation.py rather than
copied, so coupled is computed by the exact same function that produced
v2, horizon and physics -- no risk of a second, slightly different
implementation quietly drifting from the first one.

Place next to the same files run_tilt_ablation.py needs (the 8 repo
files, tilt_ablation.py, extra_metrics.py, run_tilt_ablation.py) plus
this file, then:
    python run_tilt_coupled_resume.py
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

OUT = Path("results") / "tilt_coupled_resume.json"


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

    tilt.install("coupled")
    print("\n" + "=" * 64 + "\nBRANCH coupled -- resume, all 6 folds\n" + "=" * 64)
    rows, t0 = [], time.time()
    try:
        for fold_id in range(1, len(FOLDS) + 1):
            r = evaluate_branch_fold("coupled", fold_id, X, dates, y_hard, y_soft, crisis_starts)
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
                json.dump({"branch": "coupled", "folds_done": [row["fold"] for row in rows],
                           "folds": rows}, f, indent=2)
            print(f"  [saved: {OUT}, {len(rows)}/6 folds]")
    finally:
        tilt.restore()
        CFG.SLOW_IDX, CFG.SLOW_DIM = orig_slow_idx, orig_slow_dim

    ok = [r for r in rows if r.get("roc_auc") is not None]
    if ok:
        roc = np.array([r["roc_auc"] for r in ok])
        print(f"\ncoupled: ROC-AUC mean={roc.mean():.4f} std={roc.std():.4f} "
              f"over {len(ok)}/6 folds ({time.time() - t0:.0f}s)")
    print(f"\n{OUT} now has all 6 coupled folds. Send it back along with "
          f"v2/horizon/physics (whatever file or message currently holds "
          f"them) so the four branches can be read together.")


if __name__ == "__main__":
    main()
