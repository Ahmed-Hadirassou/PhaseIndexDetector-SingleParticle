"""
run_early_stopping_test.py

Answers a different question than train_one_with_validation did: not
"does validation loss rise after some point" (already shown, twice, on
folds 2 and 5), but "does stopping there actually improve the REAL
walk-forward test ROC-AUC" -- the number that's actually reported
everywhere else, not an internal proxy for it. The two can, in
principle, disagree -- a lower validation loss doesn't automatically
mean a higher test ROC-AUC, since Hooke loss and ROC-AUC aren't the
same objective. This is the test that settles it directly rather than
inferring it from the loss curve.

Trains WITH checkpointing -- a full model snapshot saved every 5
epochs, same cadence as train_one's own training-loss log -- on the
same two folds (2 and 5) flagged before, then evaluates EACH
checkpoint's ROC-AUC on that fold's real test window. Also evaluates
the actual current model (full 45 epochs + SWA) for direct comparison
at the end.

No SWA is applied to any intermediate checkpoint (SWA only makes sense
as a final-model smoothing technique over many late epochs) -- so even
the LAST checkpoint (epoch 45) will differ slightly from the actual
current model's SWA-averaged result, printed separately as
"45_swa_current" for that reason. That's intentional: the point is raw,
per-epoch generalization, not reproducing SWA's own effect.

Verified: ZScoreNorm's fitted mean/std/clip stats are registered
buffers (main_v304_soft_labels.py), so state_dict save/load restores
them correctly -- no need to re-fit preprocessing when reconstructing
a model from a saved checkpoint below.

Place next to main_v304_soft_labels.py and walk_forward.py, then run:
    python run_early_stopping_test.py
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG, hooke_loss, PhaseIndexDetector, ChronoRegimeDataset
from walk_forward import FOLDS

FOLDS_TO_TEST = (2, 5)   # the two folds train_one_with_validation already flagged
SEEDS = (42,)
CHECKPOINT_EVERY = 5     # same cadence as train_one's own logging


def train_one_with_checkpoints(seed: int, X_tr: np.ndarray, y_tr_soft: np.ndarray, device: str,
                                persistence_tr=None, checkpoint_every: int = CHECKPOINT_EVERY):
    """Line-for-line the same training loop as main_v304_soft_labels.
    train_one (that function is never modified), plus a full model
    snapshot saved at the epoch%5==0 cadence. Returns (final_swa_model,
    checkpoints) where checkpoints is a list of (cumulative_epoch,
    state_dict) tuples -- cumulative_epoch counts across both phases
    (phase 1 ep15 = 15, phase 2 ep5 = 20, ..., phase 2 ep30 = 45)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = PhaseIndexDetector(X_tr.shape[1])
    model.fit_preprocessing(X_tr)
    model = model.to(device)

    dataset = ChronoRegimeDataset(X_tr, y_tr_soft, persistence=persistence_tr)
    loader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)

    swa_state, swa_n = None, 0
    checkpoints: list = []
    cumulative_epoch = 0

    for phase, n_epochs in (("1", 15), ("2", 30)):
        for epoch in range(1, n_epochs + 1):
            cumulative_epoch += 1
            model.train()
            s_moving = torch.zeros((CFG.BATCH_SIZE, CFG.LATENT_DIM), device=device)

            for Xt, yt, win, pers in loader:
                if Xt.shape[0] != CFG.BATCH_SIZE:
                    continue
                Xt, yt = Xt.to(device), yt.to(device)
                optimizer.zero_grad()
                if CFG.TCN_VELOCITY_ENCODER:
                    psi, _, s_moving = model(Xt, s_initial=s_moving, xf_window=win.to(device))
                else:
                    psi, _, s_moving = model(Xt, s_initial=s_moving)
                s_moving = s_moving.detach()
                loss_kwargs = {}
                if CFG.KINETIC_ENERGY_PENALTY:
                    loss_kwargs["p"] = model._last_p
                    loss_kwargs["kinetic_lambda"] = CFG.LAMBDA_KINETIC
                if CFG.DYNAMIC_K_FN:
                    loss_kwargs["persistence"] = pers.to(device)
                loss = hooke_loss(psi, yt, **loss_kwargs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            if phase == "2" and epoch >= 16:
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.detach().clone().float() for k, v in sd.items()}
                    swa_n = 1
                else:
                    swa_n += 1
                    for k, v in sd.items():
                        swa_state[k].mul_(1.0 - 1.0 / swa_n).add_(v.detach().float() / swa_n)

            if cumulative_epoch % checkpoint_every == 0:
                snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
                checkpoints.append((cumulative_epoch, snap))

    if swa_state is not None:
        target_sd = model.state_dict()
        for k in target_sd:
            target_sd[k].copy_(swa_state[k].to(target_sd[k].dtype))
        model.load_state_dict(target_sd)

    return model, checkpoints


def evaluate_checkpoints(checkpoints: list, X_test: np.ndarray, y_test_hard: np.ndarray,
                          input_dim: int, device: str) -> list:
    rows = []
    for epoch, state_dict in checkpoints:
        m = PhaseIndexDetector(input_dim).to(device)
        m.load_state_dict(state_dict)  # restores trained weights AND fitted norm stats (buffers)
        psi = mvsl.run_stateful_inference([m], X_test, device, CFG.LATENT_DIM)
        auc = float(roc_auc_score(y_test_hard, psi))
        rows.append({"epoch": epoch, "test_roc_auc": auc})
    return rows


def main() -> None:
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

    all_results = {}
    for fold_id in FOLDS_TO_TEST:
        train_end, test_start, test_end = FOLDS[fold_id - 1]
        train_mask = dates <= pd.Timestamp(train_end)
        test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
        X_tr, y_tr_soft = X[train_mask], y_soft[train_mask]
        X_te, y_te_hard = X[test_mask], y_hard[test_mask]

        print(f"\n=== fold {fold_id}: test {test_start}..{test_end} ===")
        fold_rows: list = []
        for seed in SEEDS:
            t0 = time.time()
            model, checkpoints = train_one_with_checkpoints(seed, X_tr, y_tr_soft, CFG.DEVICE)
            print(f"  seed {seed}: {len(checkpoints)} checkpoints, trained in {time.time() - t0:.0f}s")

            rows = evaluate_checkpoints(checkpoints, X_te, y_te_hard, X_tr.shape[1], CFG.DEVICE)
            for r in rows:
                r["seed"] = seed
                print(f"    epoch {r['epoch']:2d}  test_roc_auc={r['test_roc_auc']:.4f}")

            psi_swa = mvsl.run_stateful_inference([model], X_te, CFG.DEVICE, CFG.LATENT_DIM)
            swa_auc = float(roc_auc_score(y_te_hard, psi_swa))
            print(f"    [current: full 45 epochs + SWA]  test_roc_auc={swa_auc:.4f}")

            fold_rows.extend(rows)
            fold_rows.append({"epoch": "45_swa_current", "test_roc_auc": swa_auc, "seed": seed})

        best = max((r for r in fold_rows if isinstance(r["epoch"], int)), key=lambda r: r["test_roc_auc"])
        current = next(r for r in fold_rows if r["epoch"] == "45_swa_current")
        delta = best["test_roc_auc"] - current["test_roc_auc"]
        verdict = "IMPROVEMENT" if delta > 0 else ("no improvement" if delta < 0 else "tie")
        print(f"  fold {fold_id} verdict: best checkpoint = epoch {best['epoch']} "
              f"(test_roc_auc={best['test_roc_auc']:.4f}) vs current 45-epoch+SWA "
              f"({current['test_roc_auc']:.4f}) -> {verdict} of {delta:+.4f}")

        all_results[str(fold_id)] = fold_rows

    out_path = Path("results") / "early_stopping_test.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
