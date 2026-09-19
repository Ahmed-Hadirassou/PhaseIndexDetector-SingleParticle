"""
run_flickering_signals.py

Walk-forward evaluation of flicker_var and flicker_cross_rate
(flickering_signals.py), standalone and in weight-free rank ensembles
with Psi and rk_turnover (physics_signals.py's best-performing Kramers
variant). Same six folds, five seeds, same protocol as every other run
in this project.

Psi(t) is obtained from collect_physics_trajectory, already written for
physics_signals.py -- nothing new is extracted from the model. Only
the derived statistics are new.

Signals scored per fold:
    psi              reference
    flicker_var       rolling variance of Psi, 20-day window
    flicker_cross     rolling median-crossing rate of Psi, 20-day window
    rk_turnover       for reference, from physics_signals.py
    ens_psi_flkvar    rank average of psi and flicker_var
    ens_psi_rk_flk    rank average of psi, rk_turnover and flicker_var
                      (the three-mechanism ensemble: readout, escape
                      rate, and flickering, each independent)

As with physics_signals.py, standalone signals are scored in whichever
sign direction fits (a raw statistic makes no promise about direction),
and this is reported; ensembles are oriented on the calibration tail,
never on the test window.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG
from walk_forward import FOLDS, CALIB_TAIL_DAYS
from physics_signals import collect_physics_trajectory, kramers_rates
from flickering_signals import flicker_var, flicker_cross_rate, fill_lead_in
from extra_metrics import transition_auc
from run_physics_signals import _rank_avg

SEEDS = (42, 123, 456, 789, 1011)
WINDOW = 20
OUT = Path("results") / "flickering_signals.json"


def evaluate_fold(fold_id, X, dates, y_hard, y_soft, crisis_starts):
    train_end, test_start, test_end = FOLDS[fold_id - 1]
    tr = dates <= pd.Timestamp(train_end)
    te = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
    y_tr, y_te = y_hard[tr], y_hard[te]
    if y_te.sum() in (0, len(y_te)):
        return {"fold": fold_id, "note": "single-class test window"}

    models = [mvsl.train_one(s, X[tr], y_soft[tr], CFG.DEVICE)[0] for s in SEEDS]

    # Flickering needs WINDOW days of history before the test window starts,
    # or the first WINDOW-1 test days get the filled-in lead value instead of
    # a real rolling statistic. Pull those extra days from training data and
    # run inference on [lead-in days, test days] as ONE continuous sequence,
    # so the model's own state carries over correctly, then slice off the
    # lead-in portion afterward -- position-based, not boolean-mask
    # arithmetic, so there is no ambiguity about which days land where.
    train_pos, test_pos = np.where(tr)[0], np.where(te)[0]
    lead_start = max(0, train_pos[-1] - WINDOW + 2)
    ext_pos = np.concatenate([np.arange(lead_start, train_pos[-1] + 1), test_pos])
    n_lead = train_pos[-1] + 1 - lead_start

    traj_ext = collect_physics_trajectory(models, X[ext_pos], CFG.DEVICE, CFG.LATENT_DIM)
    psi_ext = traj_ext["psi"]

    fvar_ext = fill_lead_in(flicker_var(psi_ext, WINDOW))
    fcross_ext = fill_lead_in(flicker_cross_rate(psi_ext, WINDOW))
    psi_te = psi_ext[n_lead:]
    fvar_te = fvar_ext[n_lead:]
    fcross_te = fcross_ext[n_lead:]

    kr_te = kramers_rates(traj_ext["a"][n_lead:], traj_ext["b"][n_lead:],
                           traj_ext["gamma"][n_lead:], traj_ext["temp"][n_lead:])["turnover"]

    n_train = int(tr.sum())
    tail = slice(max(0, n_train - CALIB_TAIL_DAYS), n_train)
    traj_tail = collect_physics_trajectory(models, X[tr][tail], CFG.DEVICE, CFG.LATENT_DIM)
    y_tail = y_tr[tail]
    can_orient = y_tail.sum() > 0
    psi_tail = traj_tail["psi"]
    fvar_tail = fill_lead_in(flicker_var(psi_tail, WINDOW))
    kr_tail = kramers_rates(traj_tail["a"], traj_tail["b"], traj_tail["gamma"], traj_tail["temp"])["turnover"]

    scored = {"psi": psi_te, "flicker_var": fvar_te, "flicker_cross": fcross_te, "rk_turnover": kr_te}
    if can_orient:
        scored["ens_psi_flkvar"] = _rank_avg({"psi": psi_te, "flicker_var": fvar_te},
                                             {"psi": psi_tail, "flicker_var": fvar_tail}, y_tail)
        scored["ens_psi_rk_flk"] = _rank_avg(
            {"psi": psi_te, "rk_turnover": kr_te, "flicker_var": fvar_te},
            {"psi": psi_tail, "rk_turnover": kr_tail, "flicker_var": fvar_tail}, y_tail)

    out = {"fold": fold_id, "n_test": int(te.sum()), "orientable": bool(can_orient), "signals": {}}
    for name, s in scored.items():
        if name.startswith("ens_"):
            auc = roc_auc_score(y_te, s); s_used = s; flipped = False
        else:
            auc_pos = roc_auc_score(y_te, s); auc_neg = 1.0 - auc_pos
            auc = max(auc_pos, auc_neg); flipped = auc_pos < auc_neg
            s_used = s if not flipped else -s
        ta = transition_auc(s_used, dates[te], y_te, crisis_starts)["pooled"]
        out["signals"][name] = {
            "roc_auc": float(auc), "pr_auc": float(average_precision_score(y_te, s_used)),
            "transition_auc": (float(ta) if ta is not None else None), "sign_flipped": bool(flipped),
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
            for name, m in r["signals"].items():
                ta = f"{m['transition_auc']:.3f}" if m["transition_auc"] is not None else "  n/a"
                flip = " (sign flipped)" if m["sign_flipped"] else ""
                print(f"  {name:15s} ROC={m['roc_auc']:.4f}  PR={m['pr_auc']:.4f}  trans={ta}{flip}")
        OUT.parent.mkdir(exist_ok=True)
        with open(OUT, "w") as f:
            json.dump({"seeds": list(SEEDS), "window": WINDOW, "folds": rows}, f, indent=2)
        print(f"  [saved: {OUT}, {len(rows)}/{len(FOLDS)} folds, {time.time()-t0:.0f}s]")

    ok = [r for r in rows if "signals" in r]
    names = list(ok[0]["signals"].keys()) if ok else []
    print(f"\n{'signal':15s} {'ROC-AUC':>16s} {'PR-AUC':>8s} {'transAUC':>9s}")
    for name in names:
        roc = [r["signals"][name]["roc_auc"] for r in ok if name in r["signals"]]
        pr = [r["signals"][name]["pr_auc"] for r in ok if name in r["signals"]]
        ta = [r["signals"][name]["transition_auc"] for r in ok
              if name in r["signals"] and r["signals"][name]["transition_auc"] is not None]
        print(f"{name:15s} {np.mean(roc):8.4f}+-{np.std(roc):.4f} {np.mean(pr):8.4f} "
              f"{(np.mean(ta) if ta else float('nan')):9.4f}")
    print("\nCompare ens_psi_rk_flk to the two-signal ens_psi_rk already measured (physics_signals.json).")
    print("A gap there would say flickering adds a third, genuinely independent mechanism.")


if __name__ == "__main__":
    main()
