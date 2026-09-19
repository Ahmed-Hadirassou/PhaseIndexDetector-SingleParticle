"""
run_physics_signals.py

Walk-forward evaluation of every physics-derived signal in
physics_signals.py, as standalone detectors and in weight-free rank
ensembles with Psi. Same six folds, same labels, same five seeds, same
v2 architecture as the paper; nothing about the model is changed. What
is new is only what is READ from it after training.

Signals scored per fold, each as a crisis detector on the test window:

    psi              the deployed readout (reference)
    fdt              |<p^2>/d - T|, distance from equipartition
    rk_crude         gamma^-1 exp(-DV/T), the paper's current diagnostic
    rk_spatial       Kramers spatial-diffusion rate with curvature prefactors
    rk_energy        Kramers energy-diffusion (low-friction) rate
    rk_turnover      harmonic bridge across the turnover
    ens_psi_fdt      rank average of psi and fdt
    ens_psi_rk       rank average of psi and rk_turnover
    ens_all3         rank average of psi, fdt, rk_turnover

For each: ROC-AUC, PR-AUC, transition-AUC. Ensembles are oriented on
the calibration tail (last CALIB_TAIL_DAYS of the training window, the
same tail the temperature is fit on), never on the test window.

Also reported per fold: the fraction of test days in the low-friction
regime (Gamma/omega_b < 1), which says whether the paper's high-friction
assumption is a minor approximation or the wrong regime altogether.

Runtime: one full walk-forward, same as the paper's own run, plus a
few cheap inference passes. Saves after every fold.
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
from walk_forward import FOLDS, CALIB_TAIL_DAYS
from physics_signals import collect_physics_trajectory, fdt_violation, kramers_rates
from extra_metrics import transition_auc, orient

SEEDS = (42, 123, 456, 789, 1011)
OUT = Path("results") / "physics_signals.json"


def _signals(traj: dict) -> dict:
    kr = kramers_rates(traj["a"], traj["b"], traj["gamma"], traj["temp"])
    return {
        "psi": traj["psi"],
        "fdt": fdt_violation(traj["p2"], traj["temp"], CFG.LATENT_DIM),
        "rk_crude": kr["crude"], "rk_spatial": kr["spatial"],
        "rk_energy": kr["energy"], "rk_turnover": kr["turnover"],
        "_dimless_friction": kr["dimless_friction"],
    }


def _rank_avg(components_test, components_tail, y_tail):
    """Weight-free rank average, each component sign-oriented on the tail."""
    n = len(next(iter(components_test.values())))
    acc = np.zeros(n)
    for name, s_test in components_test.items():
        sgn = orient(components_tail[name], y_tail)
        acc += rankdata(sgn * s_test) / n
    return acc / len(components_test)


def evaluate_fold(fold_id, X, dates, y_hard, y_soft, crisis_starts):
    train_end, test_start, test_end = FOLDS[fold_id - 1]
    tr = dates <= pd.Timestamp(train_end)
    te = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y_hard[tr], y_hard[te]
    if y_te.sum() in (0, len(y_te)):
        return {"fold": fold_id, "note": "single-class test window"}

    models = [mvsl.train_one(s, X_tr, y_soft[tr], CFG.DEVICE)[0] for s in SEEDS]

    n_train = int(tr.sum())
    tail = slice(max(0, n_train - CALIB_TAIL_DAYS), n_train)
    sig_te = _signals(collect_physics_trajectory(models, X_te, CFG.DEVICE, CFG.LATENT_DIM))
    sig_tail = _signals(collect_physics_trajectory(models, X_tr[tail], CFG.DEVICE, CFG.LATENT_DIM))
    y_tail = y_tr[tail]
    can_orient = y_tail.sum() > 0

    dimless = sig_te.pop("_dimless_friction"); sig_tail.pop("_dimless_friction")
    scored = dict(sig_te)
    if can_orient:
        scored["ens_psi_fdt"] = _rank_avg({k: sig_te[k] for k in ("psi", "fdt")},
                                          {k: sig_tail[k] for k in ("psi", "fdt")}, y_tail)
        scored["ens_psi_rk"] = _rank_avg({k: sig_te[k] for k in ("psi", "rk_turnover")},
                                         {k: sig_tail[k] for k in ("psi", "rk_turnover")}, y_tail)
        scored["ens_all3"] = _rank_avg({k: sig_te[k] for k in ("psi", "fdt", "rk_turnover")},
                                       {k: sig_tail[k] for k in ("psi", "fdt", "rk_turnover")}, y_tail)

    out = {"fold": fold_id, "n_test": int(te.sum()),
           "low_friction_fraction": float((dimless < 1.0).mean()),
           "orientable": bool(can_orient), "signals": {}}
    for name, s in scored.items():
        # standalone signals are scored both ways and the better direction
        # is reported, because a raw physics quantity carries no promise
        # about its sign; ensembles are already oriented on the tail.
        if name.startswith("ens_"):
            auc = roc_auc_score(y_te, s); s_used = s
        else:
            auc_pos = roc_auc_score(y_te, s); auc_neg = 1.0 - auc_pos
            auc = max(auc_pos, auc_neg); s_used = s if auc_pos >= auc_neg else -s
        ta = transition_auc(s_used, dates[te], y_te, crisis_starts)["pooled"]
        out["signals"][name] = {
            "roc_auc": float(auc),
            "pr_auc": float(average_precision_score(y_te, s_used)),
            "transition_auc": (float(ta) if ta is not None else None),
            "sign_flipped": (not name.startswith("ens_")) and bool(auc_pos < auc_neg),
        }
    return out


def main() -> None:
    print("Loading price data and building features...")
    prices, all_prices = mvsl.load_price_data(CFG)
    returns, features = mvsl.build_feature_matrix(prices, all_prices)
    hard = mvsl.make_hard_labels(features.index)
    soft = mvsl.build_soft_labels(features.index)
    common = features.index.intersection(hard.index)
    X = features.loc[common].values.astype(np.float32)
    y_hard = hard.loc[common].values
    y_soft = soft.loc[common].values.astype(np.float32)
    dates = common
    crisis_starts = [s for s, _ in mvsl.CRISES]

    rows, t0 = [], time.time()
    for fold_id in range(1, len(FOLDS) + 1):
        print(f"\n=== fold {fold_id} ===")
        r = evaluate_fold(fold_id, X, dates, y_hard, y_soft, crisis_starts)
        rows.append(r)
        if "signals" in r:
            print(f"  low-friction days: {r['low_friction_fraction']*100:.0f}%   "
                  f"ensembles oriented: {r['orientable']}")
            for name, m in r["signals"].items():
                ta = f"{m['transition_auc']:.3f}" if m["transition_auc"] is not None else "  n/a"
                flip = " (sign flipped)" if m["sign_flipped"] else ""
                print(f"  {name:12s} ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  trans={ta}{flip}")
        OUT.parent.mkdir(exist_ok=True)
        with open(OUT, "w") as f:
            json.dump({"seeds": list(SEEDS), "folds": rows}, f, indent=2)
        print(f"  [saved: {OUT}, {len(rows)}/{len(FOLDS)} folds, {time.time()-t0:.0f}s]")

    ok = [r for r in rows if "signals" in r]
    names = list(ok[0]["signals"].keys()) if ok else []
    print(f"\n{'signal':12s} {'ROC-AUC':>16s} {'PR-AUC':>8s} {'transAUC':>9s}  (mean over {len(ok)} folds)")
    for name in names:
        roc = [r["signals"][name]["roc_auc"] for r in ok if name in r["signals"]]
        pr = [r["signals"][name]["pr_auc"] for r in ok if name in r["signals"]]
        ta = [r["signals"][name]["transition_auc"] for r in ok
              if name in r["signals"] and r["signals"][name]["transition_auc"] is not None]
        print(f"{name:12s} {np.mean(roc):8.4f}+-{np.std(roc):.4f} {np.mean(pr):8.4f} "
              f"{(np.mean(ta) if ta else float('nan')):9.4f}")
    lf = [r["low_friction_fraction"] for r in ok]
    print(f"\nlow-friction (Gamma/omega_b < 1) share of test days, mean over folds: {np.mean(lf)*100:.0f}%")
    print("If that number is large, the paper's high-friction Kramers assumption is the")
    print("wrong regime for a substantial part of the sample, not a minor approximation.")


if __name__ == "__main__":
    main()
