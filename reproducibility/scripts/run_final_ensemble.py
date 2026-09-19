"""
run_final_ensemble.py

The one experiment that combines what this project has actually
established, without adding anything:

  ARCHITECTURE  LF -- learned potential, fixed thermostat. Of the four
                mechanism branches it had the best calibration (ECE
                0.096), the best fold 4, PR-AUC and F1 equal to the
                full model, and 2.2x lower seed variance. It keeps the
                learned potential the paper is about and drops the two
                thermostat heads the 2x2 showed to be unnecessary.

  READOUT       weight-free rank ensemble of Psi with the Kramers rate
                (the +0.030 measured earlier), and the same with
                flicker_var as a third component.

  ORIENTATION   on the FULL training window instead of the 504-day
                calibration tail. The tail holds zero crisis days on
                folds 1, 3 and 4, which is why every ensemble result so
                far was a 3-fold result. The full training window holds
                500+ crisis days on every fold. Orientation is one bit
                per signal, not a temperature; it does not need to be
                recent, it needs to be stable -- and both signs were
                measured stable across all six folds (rk flipped 6/6,
                flicker_var 0/6). No test information is used.

LL is run alongside with the identical readout so the LF-vs-LL
difference is measured under the same ensemble, not inferred from
earlier 3-fold numbers.

Both branches, 6 folds, 5 seeds, per-seed scores, saves after every
fold. Roughly two hours. Output: results/final_ensemble.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG
from walk_forward import FOLDS
import tilt_ablation as tilt
import mechanism_ablation as mech
from physics_signals import collect_physics_trajectory, kramers_rates
from flickering_signals import flicker_var, fill_lead_in
from extra_metrics import transition_auc, orient

SEEDS = (42, 123, 456, 789, 1011)
BRANCHES = ("LF", "LL")
WINDOW = 20
OUT = Path("results") / "final_ensemble.json"


def _signals(traj):
    kr = kramers_rates(traj["a"], traj["b"], traj["gamma"], traj["temp"])["turnover"]
    return {"psi": traj["psi"], "rk": kr, "flk": fill_lead_in(flicker_var(traj["psi"], WINDOW))}


def _rank_avg(test, train, y_train, keys):
    n = len(test["psi"]); acc = np.zeros(n)
    for k in keys:
        acc += rankdata(orient(train[k], y_train) * test[k]) / n
    return acc / len(keys)


def evaluate(mode, fold_id, X, dates, y_hard, y_soft, crisis_starts):
    train_end, test_start, test_end = FOLDS[fold_id - 1]
    tr = dates <= pd.Timestamp(train_end)
    te = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
    y_tr, y_te = y_hard[tr], y_hard[te]
    if y_te.sum() in (0, len(y_te)):
        return {"fold": fold_id, "note": "single-class test window"}

    models, per_seed = [], []
    for s in SEEDS:
        m, _ = mvsl.train_one(s, X[tr], y_soft[tr], CFG.DEVICE)
        models.append(m)
        per_seed.append(float(roc_auc_score(y_te, mvsl.run_stateful_inference([m], X[te], CFG.DEVICE, CFG.LATENT_DIM))))

    # test window with a WINDOW-day lead-in from training so flicker_var is
    # a real rolling statistic from the first test day (position-based)
    train_pos, test_pos = np.where(tr)[0], np.where(te)[0]
    lead_start = max(0, train_pos[-1] - WINDOW + 2)
    ext_pos = np.concatenate([np.arange(lead_start, train_pos[-1] + 1), test_pos])
    n_lead = train_pos[-1] + 1 - lead_start
    sig_ext = _signals(collect_physics_trajectory(models, X[ext_pos], CFG.DEVICE, CFG.LATENT_DIM))
    sig_te = {k: v[n_lead:] for k, v in sig_ext.items()}

    # orientation on the FULL training window
    sig_tr = _signals(collect_physics_trajectory(models, X[tr], CFG.DEVICE, CFG.LATENT_DIM))

    scored = {
        "psi": sig_te["psi"],
        "ens2_psi_rk": _rank_avg(sig_te, sig_tr, y_tr, ("psi", "rk")),
        "ens3_psi_rk_flk": _rank_avg(sig_te, sig_tr, y_tr, ("psi", "rk", "flk")),
    }
    out = {"fold": fold_id, "n_test": int(te.sum()), "per_seed_psi_roc": per_seed,
           "seed_std": float(np.std(per_seed)),
           "orientation_signs": {k: float(orient(sig_tr[k], y_tr)) for k in ("rk", "flk")},
           "signals": {}}
    for name, s in scored.items():
        ta = transition_auc(s, dates[te], y_te, crisis_starts)["pooled"]
        out["signals"][name] = {"roc_auc": float(roc_auc_score(y_te, s)),
                                "pr_auc": float(average_precision_score(y_te, s)),
                                "transition_auc": (float(ta) if ta is not None else None)}
    return out


def main():
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
    orig = (list(CFG.SLOW_IDX), CFG.SLOW_DIM)
    CFG.SLOW_IDX, CFG.SLOW_DIM = slow_idx, slow_dim
    crisis_starts = [s for s, _ in mvsl.CRISES]

    results, t0 = {}, time.time()
    try:
        for mode in BRANCHES:
            mech.install(mode)
            print(f"\n{'=' * 64}\nBRANCH {mode}\n{'=' * 64}")
            rows = []
            for fold_id in range(1, len(FOLDS) + 1):
                r = evaluate(mode, fold_id, X, dates, y_hard, y_soft, crisis_starts)
                rows.append(r)
                if "signals" in r:
                    print(f"  fold {fold_id}: seed_sd={r['seed_std']:.4f}  signs={r['orientation_signs']}")
                    for n, m in r["signals"].items():
                        ta = f"{m['transition_auc']:.3f}" if m["transition_auc"] is not None else "n/a"
                        print(f"    {n:16s} ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  trans={ta}")
                results[mode] = rows
                OUT.parent.mkdir(exist_ok=True)
                with open(OUT, "w") as f:
                    json.dump({"seeds": list(SEEDS), "window": WINDOW, "branches": results}, f, indent=2)
                print(f"    [saved, {len(rows)}/6 folds of {mode}, {time.time() - t0:.0f}s]")
    finally:
        mech.restore()
        CFG.SLOW_IDX, CFG.SLOW_DIM = orig

    print(f"\n{'=' * 64}\nSUMMARY -- all 6 folds, no fold excluded\n{'=' * 64}")
    print(f"{'branch':6s} {'signal':16s} {'ROC-AUC':>16s} {'PR-AUC':>8s} {'transAUC':>9s}")
    for mode, rows in results.items():
        ok = [r for r in rows if "signals" in r]
        for n in ("psi", "ens2_psi_rk", "ens3_psi_rk_flk"):
            roc = [r["signals"][n]["roc_auc"] for r in ok]
            pr = [r["signals"][n]["pr_auc"] for r in ok]
            ta = [r["signals"][n]["transition_auc"] for r in ok if r["signals"][n]["transition_auc"] is not None]
            print(f"{mode:6s} {n:16s} {np.mean(roc):8.4f}+-{np.std(roc):.4f} {np.mean(pr):8.4f} {np.mean(ta):9.4f}")
    print("\nThe number that matters: ens2_psi_rk on LF, 6 folds, against psi on LF, 6 folds,")
    print("read fold by fold against the seed_std column as always.")


if __name__ == "__main__":
    main()
