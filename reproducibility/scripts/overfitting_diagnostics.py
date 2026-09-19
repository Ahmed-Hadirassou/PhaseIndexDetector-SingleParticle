"""
overfitting_diagnostics.py

Four tools, in order of cost (cheapest first):

1. train_test_auc_gap
   No retraining. Reuses an ensemble you already trained. Compares
   in-sample (training window) vs out-of-sample (test window) ROC-AUC.
   The classic, cheapest overfitting signal: a large gap (train >> test)
   is itself evidence, even with nothing fancier.

2. train_one_with_validation
   A full COPY of main_v304_soft_labels.train_one (your real train_one
   is completely untouched), with one addition: at the same epoch%5==0
   cadence as the existing training-loss log, it also runs the live
   (non-SWA) model on a held-out validation slice via the same
   run_stateful_inference() used everywhere else in this codebase, and
   logs hooke_loss on that output. This is the diagnostic
   plot_training_curves() in visualization.py does NOT currently give
   you -- that plot is per-seed TRAINING loss only, there is no
   validation split tracked anywhere in the pipeline today. X_val /
   y_val_soft must be carved from INSIDE the training window (e.g. the
   last CALIB_TAIL_DAYS) -- never from the real walk-forward test set,
   or this stops being a training diagnostic and becomes a second leak.

3. permutation_test
   The actual formal statistical test. Randomly repositions the 13
   CRISES windows -- preserving each window's LENGTH and rejecting
   overlaps, i.e. a BLOCK permutation, not a naive day-by-day label
   shuffle. A day-by-day shuffle would destroy the contiguous,
   autocorrelated block structure real crises have, making the null
   distribution far too easy to beat and overstating significance.
   Rebuilds labels, retrains, evaluates walk-forward ROC-AUC on the
   permuted labels, repeats n_permutations times, and reports where the
   REAL result sits in that empirical null -> an actual p-value, not a
   heuristic.

4. hyperparameter_sensitivity_sweep
   Small grid around LATENT_DIM / N_VERLET_STEPS, rerun on a subset of
   folds. Does not prove absence of leakage -- checks a different thing:
   whether performance is a narrow spike at exactly the current values
   (fragile, possibly cherry-picked) or holds up across a neighborhood.

Compute note, same for #3 and #4: both multiply your existing per-fold
training time (visible in your own Colab logs as "ensemble trained in
Xs") by however many permutations / grid points you run. Defaults below
are deliberately small (single seed, 4 folds) to make a first run
tractable; widen once you know your own per-run cost.

Place this file next to main_v304_soft_labels.py and walk_forward.py.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import main_v304_soft_labels as mvsl
from main_v304_soft_labels import CFG, hooke_loss
from walk_forward import FOLDS


# ---------------------------------------------------------------------------
# 1. In-sample vs out-of-sample AUC gap -- no retraining needed
# ---------------------------------------------------------------------------

def train_test_auc_gap(models: list, X_train: np.ndarray, y_train_hard: np.ndarray,
                        X_test: np.ndarray, y_test_hard: np.ndarray,
                        device: str, latent_dim: int) -> dict:
    """Call this right after a fold's ensemble is trained -- e.g. paste
    into evaluate_fold() in walk_forward.py right after `models` is
    built, or call standalone with models you already have saved."""
    psi_train = mvsl.run_stateful_inference(models, X_train, device, latent_dim)
    psi_test = mvsl.run_stateful_inference(models, X_test, device, latent_dim)
    train_auc = roc_auc_score(y_train_hard, psi_train)
    test_auc = roc_auc_score(y_test_hard, psi_test)
    return {"train_auc": train_auc, "test_auc": test_auc, "gap": train_auc - test_auc}


# ---------------------------------------------------------------------------
# 2. Train / validation loss curves
# ---------------------------------------------------------------------------

def train_one_with_validation(seed: int, X_tr: np.ndarray, y_tr_soft: np.ndarray,
                               X_val: np.ndarray, y_val_soft: np.ndarray,
                               device: str, persistence_tr=None, swa_start_epoch: int = 16,
                               verbose: bool = True):
    """Line-for-line the same training loop as main_v304_soft_labels.
    train_one (your original is never modified), plus a validation-loss
    read-out at the same 5-epoch cadence. Note: the validation loss below
    skips the KINETIC_ENERGY_PENALTY / DYNAMIC_K_FN loss_kwargs that the
    TRAINING loss can include when those flags are on -- a deliberate
    simplification so val loss stays a plain, comparable hooke_loss
    number; if you have either flag on, treat the two curves' absolute
    levels as not perfectly apples-to-apples, but their SHAPE over epochs
    (does val start rising while train keeps falling) is still valid.

    Returns (model, train_loss_history, val_loss_history).
    """
    import random
    from torch import optim
    from torch.utils.data import DataLoader
    from main_v304_soft_labels import PhaseIndexDetector, ChronoRegimeDataset

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
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []

    y_val_t = torch.tensor(y_val_soft, dtype=torch.float32).to(device)

    for phase, n_epochs in (("1", 15), ("2", 30)):
        for epoch in range(1, n_epochs + 1):
            model.train()
            total_loss = 0.0
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
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(loader)

            if phase == "2" and epoch >= swa_start_epoch:
                sd = model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.detach().clone().float() for k, v in sd.items()}
                    swa_n = 1
                else:
                    swa_n += 1
                    for k, v in sd.items():
                        swa_state[k].mul_(1.0 - 1.0 / swa_n).add_(v.detach().float() / swa_n)

            if epoch % 5 == 0:
                train_loss_history.append(avg_loss)
                with torch.no_grad():
                    model.eval()
                    psi_val = mvsl.run_stateful_inference([model], X_val, device, CFG.LATENT_DIM)
                    val_loss = hooke_loss(torch.tensor(psi_val, dtype=torch.float32).to(device),
                                           y_val_t).item()
                    val_loss_history.append(val_loss)
                    model.train()
                if verbose:
                    print(f"seed {seed} | phase {phase} ep {epoch:2d} | "
                          f"train_loss={avg_loss:.4f} | val_loss={val_loss:.4f}")

    if swa_state is not None:
        target_sd = model.state_dict()
        for k in target_sd:
            target_sd[k].copy_(swa_state[k].to(target_sd[k].dtype))
        model.load_state_dict(target_sd)

    return model, train_loss_history, val_loss_history


# ---------------------------------------------------------------------------
# 3. Permutation test
# ---------------------------------------------------------------------------

def _random_block_permutation(crises: list, valid_start, valid_end,
                               rng: np.random.Generator, max_tries: int = 200) -> list:
    """Repositions each (start, end) window to a random new start,
    keeping each window's length fixed, rejecting overlaps among the
    repositioned windows. Block permutation, not i.i.d. day shuffling --
    see module docstring for why that distinction matters here."""
    valid_start, valid_end = pd.Timestamp(valid_start), pd.Timestamp(valid_end)
    total_days = (valid_end - valid_start).days

    for _ in range(max_tries):
        new_windows = []
        for start, end in crises:
            length = (pd.Timestamp(end) - pd.Timestamp(start)).days
            offset = rng.integers(0, max(1, total_days - length))
            new_start = valid_start + pd.Timedelta(days=int(offset))
            new_end = new_start + pd.Timedelta(days=length)
            new_windows.append((new_start, new_end))
        new_windows.sort()
        if all(new_windows[i][1] < new_windows[i + 1][0] for i in range(len(new_windows) - 1)):
            return [(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in new_windows]
    raise RuntimeError("could not find a non-overlapping permutation within max_tries; "
                        "raise max_tries or check for an unusually long crisis window")


def permutation_test(X: np.ndarray, dates: pd.DatetimeIndex, fold_ids: tuple = (1, 2, 3, 4),
                      n_permutations: int = 50, seeds: tuple = (42,), device: str = None,
                      random_state: int = 0) -> dict:
    """X must be the FULL feature matrix spanning all dates (same X built
    in walk_forward.py's main()), not pre-sliced to any one fold.

    fold_ids defaults to (1,2,3,4) purely to keep compute tractable (4
    folds, 1 seed, instead of 6 folds x 5 seeds) -- this is unrelated to
    the separate fold-5/6-vs-1-4 discussion about hyperparameter
    provenance; any subset of FOLDS is a methodologically valid choice
    for THIS test, widen it once you know the per-permutation cost.

    Returns real_mean_auc, the null distribution, its mean/std, and an
    empirical p-value = P(null_auc >= real_mean_auc).
    """
    device = device or CFG.DEVICE
    rng = np.random.default_rng(random_state)
    original_crises = list(mvsl.CRISES)

    def _fold_aucs_for_current_crises():
        y_hard = mvsl.make_hard_labels(dates)
        y_soft = mvsl.build_soft_labels(dates)
        aucs = []
        for fold_id in fold_ids:
            train_end, test_start, test_end = FOLDS[fold_id - 1]
            train_mask = dates <= pd.Timestamp(train_end)
            test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
            if y_hard.values[test_mask].sum() == 0 or y_hard.values[test_mask].sum() == test_mask.sum():
                continue  # single-class window, e.g. after an unlucky permutation
            models = [mvsl.train_one(s, X[train_mask], y_soft.values[train_mask], device)[0]
                      for s in seeds]
            psi_test = mvsl.run_stateful_inference(models, X[test_mask], device, CFG.LATENT_DIM)
            aucs.append(roc_auc_score(y_hard.values[test_mask], psi_test))
        return aucs

    real_aucs = _fold_aucs_for_current_crises()
    real_mean_auc = float(np.mean(real_aucs))
    print(f"Real labels: mean ROC-AUC over folds {fold_ids} = {real_mean_auc:.4f}")

    null_aucs = []
    t0 = time.time()
    for p in range(n_permutations):
        permuted = _random_block_permutation(original_crises, dates.min(), dates.max(), rng)
        mvsl.CRISES = permuted
        try:
            fold_aucs = _fold_aucs_for_current_crises()
            if fold_aucs:
                null_aucs.append(float(np.mean(fold_aucs)))
        finally:
            mvsl.CRISES = original_crises  # always restore, even if this permutation errors
        elapsed = time.time() - t0
        print(f"  permutation {p + 1}/{n_permutations}  "
              f"null_auc={null_aucs[-1] if null_aucs else float('nan'):.4f}  "
              f"({elapsed:.0f}s elapsed, ~{elapsed / (p + 1):.0f}s/permutation)")

    null_aucs = np.array(null_aucs)
    p_value = float((null_aucs >= real_mean_auc).mean()) if len(null_aucs) else float("nan")
    return {
        "real_mean_auc": real_mean_auc,
        "real_fold_aucs": real_aucs,
        "null_aucs": null_aucs.tolist(),
        "null_mean": float(null_aucs.mean()) if len(null_aucs) else None,
        "null_std": float(null_aucs.std()) if len(null_aucs) else None,
        "empirical_p_value": p_value,
        "n_permutations_completed": len(null_aucs),
    }


# ---------------------------------------------------------------------------
# 4. Hyperparameter sensitivity sweep
# ---------------------------------------------------------------------------

def hyperparameter_sensitivity_sweep(X: np.ndarray, dates: pd.DatetimeIndex,
                                      latent_dims: tuple = (3, 4, 5),
                                      verlet_steps: tuple = (4, 6, 8),
                                      fold_ids: tuple = (1, 2, 3, 4),
                                      seeds: tuple = (42,), device: str = None) -> pd.DataFrame:
    """Reruns walk-forward on a small grid around the current
    (LATENT_DIM=4, N_VERLET_STEPS=6). Relies on PhaseIndexDetector
    reading CFG.LATENT_DIM / CFG.N_VERLET_STEPS fresh at construction
    time, which matches this codebase's established pattern (CFG.* is
    referenced directly throughout, not cached at import time) -- worth
    a quick sanity check on your end if a result looks off.
    CFG is restored to its original values in `finally`, whatever
    happens during the sweep. 3x3=9 configs x len(fold_ids) folds x
    len(seeds) seeds -- same compute-scaling caveat as permutation_test.
    """
    device = device or CFG.DEVICE
    orig_latent, orig_verlet = CFG.LATENT_DIM, CFG.N_VERLET_STEPS
    y_hard = mvsl.make_hard_labels(dates)
    y_soft = mvsl.build_soft_labels(dates)

    rows = []
    try:
        for ld in latent_dims:
            for nv in verlet_steps:
                CFG.LATENT_DIM, CFG.N_VERLET_STEPS = ld, nv
                fold_aucs = []
                for fold_id in fold_ids:
                    train_end, test_start, test_end = FOLDS[fold_id - 1]
                    train_mask = dates <= pd.Timestamp(train_end)
                    test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
                    models = [mvsl.train_one(s, X[train_mask], y_soft.values[train_mask], device)[0]
                              for s in seeds]
                    psi_test = mvsl.run_stateful_inference(models, X[test_mask], device, CFG.LATENT_DIM)
                    fold_aucs.append(roc_auc_score(y_hard.values[test_mask], psi_test))
                mean_auc = float(np.mean(fold_aucs))
                is_current = (ld == orig_latent and nv == orig_verlet)
                print(f"  LATENT_DIM={ld}  N_VERLET_STEPS={nv}  mean_auc={mean_auc:.4f}"
                      f"{'  <-- current config' if is_current else ''}")
                rows.append({"latent_dim": ld, "n_verlet_steps": nv, "mean_roc_auc": mean_auc,
                             "is_current_config": is_current})
    finally:
        CFG.LATENT_DIM, CFG.N_VERLET_STEPS = orig_latent, orig_verlet

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Learning curve vs. training-set size
# ---------------------------------------------------------------------------

def learning_curve_vs_sample_size(X: np.ndarray, dates: pd.DatetimeIndex, y_hard: np.ndarray,
                                   y_soft: np.ndarray, fold_id: int,
                                   fractions: tuple = (0.5, 0.75, 1.0), seeds: tuple = (42,),
                                   device: str = None) -> pd.DataFrame:
    """Different question from every test above: not 'is the signal real'
    or 'does the model memorize', but 'is performance data-starved'. Trains
    on the LAST {fraction} of the fold's training window (truncating the
    START, keeping train_end fixed -- still strictly causal) and checks
    whether test performance keeps climbing as more history is added.
    Climbing -> the model is using the extra data, consistent with
    genuine generalization rather than having already saturated on a
    small, potentially-memorized slice. Flat or declining -> more data
    isn't helping, worth a closer look.

    X, dates, y_hard, y_soft must be the FULL arrays spanning all dates
    (same ones built in walk_forward.py's main()), not pre-sliced to a
    fold.
    """
    device = device or CFG.DEVICE
    train_end, test_start, test_end = FOLDS[fold_id - 1]
    full_train_mask = dates <= pd.Timestamp(train_end)
    test_mask = (dates >= pd.Timestamp(test_start)) & (dates <= pd.Timestamp(test_end))
    train_positions = np.where(full_train_mask)[0]
    n_train_full = len(train_positions)

    rows = []
    for frac in fractions:
        n_use = max(int(n_train_full * frac), 100)
        use_positions = train_positions[-n_use:]
        X_tr, y_tr_soft = X[use_positions], y_soft[use_positions]
        y_tr_hard = y_hard[use_positions]
        if y_tr_hard.sum() == 0:
            print(f"  fraction={frac:.2f}  n_train={n_use}  SKIPPED (0 crisis days at this size)")
            continue
        models = [mvsl.train_one(s, X_tr, y_tr_soft, device)[0] for s in seeds]
        psi_test = mvsl.run_stateful_inference(models, X[test_mask], device, CFG.LATENT_DIM)
        auc = float(roc_auc_score(y_hard[test_mask], psi_test))
        rows.append({"fraction": frac, "n_train": int(n_use), "roc_auc": auc})
        print(f"  fraction={frac:.2f}  n_train={n_use:5d}  test_roc_auc={auc:.4f}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Feature-group ablation
# ---------------------------------------------------------------------------

def feature_group_ablation(prices, all_prices, dates: pd.DatetimeIndex, fold_id: int,
                            groups_to_test: list = None, seeds: tuple = (42,),
                            device: str = None) -> pd.DataFrame:
    """Different question again: not memorization, not signal-vs-noise,
    not data scaling -- whether the signal depends narrowly on ONE
    feature source or is robust across them.

    IMPORTANT, found by actually running this against the real model
    before shipping it: CFG.SLOW_IDX / CFG.FAST_IDX are ABSOLUTE column
    positions hardcoded for the standard 204-column layout (statistical
    + macro = "slow", volatility + momentum + market + velocity =
    "fast"). Simply dropping a group and re-concatenating the rest
    shifts every later column's position and silently breaks that
    split (an early version of this function did exactly that and
    crashed with an out-of-bounds index on the first real test). So
    this ablates by REPLACING a group's columns with their own
    training-period mean -- informationally flat, zero variance for the
    model to use -- while keeping all 204 columns in their original
    positions, which keeps SLOW_IDX/FAST_IDX valid. Same functions
    build_feature_matrix itself uses (build_base_features,
    build_velocity_features), same concatenation order -- not a
    separately-reimplemented feature set that could drift from the
    real one.

    prices, all_prices: exactly what mvsl.load_price_data(mvsl.CFG)
    returns -- pass those two return values straight through.
    dates: the SAME date index used elsewhere (e.g. the `common` index
    from walk_forward.py's main()).

    Retrains from scratch per group (7 trainings total: full + 6
    ablations) -- the expensive one of the six tests here. Defaults to
    a single fold and single seed; widen once you know the per-training
    cost from your own logs.
    """
    device = device or CFG.DEVICE
    train_end, test_start, test_end = FOLDS[fold_id - 1]

    returns, stat, vol, mom, mkt, mac = mvsl.build_base_features(prices, all_prices)
    velocity = mvsl.build_velocity_features(returns)
    ordered_names = ["statistical", "volatility", "momentum", "market", "macro", "velocity"]
    ordered_frames = [f.reindex(returns.index) for f in (stat, vol, mom, mkt, mac, velocity)]
    full_feats = pd.concat(ordered_frames, axis=1).ffill().bfill().fillna(0)
    common = full_feats.index.intersection(dates)
    X_full = full_feats.loc[common].values.astype(np.float32)

    bounds, start = {}, 0
    for name, frame in zip(ordered_names, ordered_frames):
        width = frame.shape[1]
        bounds[name] = (start, start + width)
        start += width

    y_hard_c = mvsl.make_hard_labels(common).values
    y_soft_c = mvsl.build_soft_labels(common).values.astype(np.float32)
    train_mask = common <= pd.Timestamp(train_end)
    test_mask = (common >= pd.Timestamp(test_start)) & (common <= pd.Timestamp(test_end))
    groups_to_test = groups_to_test or ordered_names

    def _ablated_X(exclude: str = None) -> np.ndarray:
        Xc = X_full.copy()
        if exclude is not None:
            lo, hi = bounds[exclude]
            Xc[:, lo:hi] = Xc[train_mask, lo:hi].mean(axis=0)
        return Xc

    def _train_and_score(exclude) -> float:
        Xc = _ablated_X(exclude)
        models = [mvsl.train_one(s, Xc[train_mask], y_soft_c[train_mask], device)[0] for s in seeds]
        psi_test = mvsl.run_stateful_inference(models, Xc[test_mask], device, CFG.LATENT_DIM)
        return float(roc_auc_score(y_hard_c[test_mask], psi_test))

    rows = []
    auc_full = _train_and_score(exclude=None)
    rows.append({"excluded": "none (full model)", "roc_auc": auc_full, "drop_from_full": 0.0})
    print(f"  excluded=none (full)        test_roc_auc={auc_full:.4f}")

    for name in groups_to_test:
        auc = _train_and_score(exclude=name)
        rows.append({"excluded": name, "roc_auc": auc, "drop_from_full": auc_full - auc})
        print(f"  excluded={name:12s}  test_roc_auc={auc:.4f}  (drop={auc_full - auc:+.4f})")

    return pd.DataFrame(rows)
