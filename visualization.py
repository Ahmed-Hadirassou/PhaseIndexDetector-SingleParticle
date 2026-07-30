"""
Reporting figures for the PhaseIndex detector.

Every function here takes arrays that have already been computed elsewhere
and either returns a Matplotlib Figure (save_path=None) or saves a PNG and
returns its path. Nothing in this module runs the model or touches the
training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

COLOR_STABLE = "#3B6E8F"
COLOR_CRISIS = "#C1440E"
COLOR_CALIB = "#4C9A6A"
COLOR_BASE = "#8C8C8C"
COLOR_ACCENT = "#B08D57"
PALETTE = [COLOR_STABLE, COLOR_CRISIS, COLOR_ACCENT, COLOR_CALIB]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.grid": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "legend.frameon": False,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def _save_or_return(fig, save_path):
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return str(save_path)
    return fig


# ---------------------------------------------------------------------------
# Detection quality
# ---------------------------------------------------------------------------

def plot_roc_pr(y_true, scores, save_path=None):
    apply_style()
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    base_rate = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.plot(fpr, tpr, color=COLOR_CRISIS, lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=COLOR_BASE, lw=1, ls="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(rec, prec, color=COLOR_CRISIS, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
    ax.axhline(base_rate, color=COLOR_BASE, lw=1, ls="--", label=f"Base rate = {base_rate:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_calibration(y_true, y_prob, n_bins=10, save_path=None):
    apply_style()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    obs, pred, counts = [], [], []
    for b in range(n_bins):
        m = bin_ids == b
        if not m.any():
            continue
        obs.append(y_true[m].mean())
        pred.append(y_prob[m].mean())
        counts.append(int(m.sum()))

    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], color=COLOR_BASE, lw=1, ls="--", label="Perfect calibration")
    ax.plot(pred, obs, "o-", color=COLOR_CRISIS, lw=2, ms=6, label="Model")
    ax_inset = ax.inset_axes([0.08, 0.70, 0.36, 0.24])
    ax_inset.bar(pred, counts, width=0.05, color=COLOR_ACCENT)
    ax_inset.set_title("Bin counts", fontsize=8)
    ax_inset.tick_params(labelsize=7)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_confusion_matrices(cm_base, cm_calib, labels=("Stable", "Crisis"), save_path=None):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, cm, title in zip(axes, [np.asarray(cm_base), np.asarray(cm_calib)],
                              ["Base radar", "Calibrated radar"]):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title)
        ax.grid(False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_threshold_sweep(thresholds, recalls, precisions, f1s, tau_star=None, save_path=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(thresholds, recalls, "o-", color=COLOR_CRISIS, label="Recall")
    ax.plot(thresholds, precisions, "o-", color=COLOR_STABLE, label="Precision")
    ax.plot(thresholds, f1s, "o-", color=COLOR_ACCENT, label="F1")
    if tau_star is not None:
        ax.axvline(tau_star, color="#555555", lw=1, ls="--", label=f"tau* = {tau_star:.2f}")
    ax.set_xlabel("Decision threshold (tau)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep")
    ax.legend()
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Phase index behavior
# ---------------------------------------------------------------------------

def plot_psi_timeseries(dates, psi_scaled, y_true_hard, crisis_windows=None,
                         threshold=None, save_path=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, psi_scaled, color=COLOR_STABLE, lw=1.1, label="Phase index (Psi)")
    y_true_hard = np.asarray(y_true_hard)
    ax.fill_between(dates, 0, psi_scaled, where=(y_true_hard == 1),
                     color=COLOR_CRISIS, alpha=0.18, label="Labeled crisis")
    if threshold is not None:
        ax.axhline(threshold, color=COLOR_ACCENT, lw=1, ls="--", label=f"Threshold = {threshold:.0f}")
    if crisis_windows:
        ymax = max(psi_scaled.max(), threshold or 0) * 1.05
        for name, start, end in crisis_windows:
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ax.annotate(name, xy=(mid, ymax), ha="center", fontsize=8.5, color=COLOR_CRISIS)
    ax.set_ylabel("Psi (0-100)")
    ax.set_title("Phase index over the test window")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_psi_distribution(scores, y_true, cohens_d_value=None, save_path=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    bins = np.linspace(0, 1, 41)
    y_true = np.asarray(y_true)
    ax.hist(scores[y_true == 0], bins=bins, color=COLOR_STABLE, alpha=0.65,
            density=True, label="Stable days")
    ax.hist(scores[y_true == 1], bins=bins, color=COLOR_CRISIS, alpha=0.65,
            density=True, label="Crisis days")
    ax.set_xlabel("Psi")
    ax.set_ylabel("Density")
    title = "Phase index distribution by regime"
    if cohens_d_value is not None:
        title += f"  (Cohen's d = {cohens_d_value:.2f})"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_physics_correlations(gamma, temp, alpha, psi, save_path=None):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    ax = axes[0]
    sc = ax.scatter(temp, psi, c=gamma, cmap="coolwarm", s=10, alpha=0.6)
    ax.set_xlabel("Temperature (T)")
    ax.set_ylabel("Psi")
    ax.set_title("Psi vs. temperature (color = friction, gamma)")
    fig.colorbar(sc, ax=ax, label="gamma")

    ax = axes[1]
    labels = ["gamma vs Psi", "T vs Psi"]
    corrs = [float(np.corrcoef(gamma, psi)[0, 1]), float(np.corrcoef(temp, psi)[0, 1])]
    colors = [COLOR_STABLE if c < 0 else COLOR_CRISIS for c in corrs]
    ax.barh(labels, corrs, color=colors)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title(f"Correlations with Psi  (mean alpha = {np.mean(alpha):.3f})")
    for i, c in enumerate(corrs):
        ax.text(c + (0.03 if c >= 0 else -0.03), i, f"{c:+.2f}",
                va="center", ha="left" if c >= 0 else "right", fontsize=9)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Statistical rigor
# ---------------------------------------------------------------------------

def plot_bootstrap_distribution(samples, point_estimate, reference_value=None,
                                 reference_label="V30.3", metric_name="ROC-AUC", save_path=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.hist(samples, bins=40, color=COLOR_STABLE, alpha=0.75)
    ax.axvline(point_estimate, color=COLOR_CRISIS, lw=2, label=f"Point estimate = {point_estimate:.4f}")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    ax.axvspan(lo, hi, color=COLOR_CRISIS, alpha=0.08, label=f"95% CI [{lo:.4f}, {hi:.4f}]")
    if reference_value is not None:
        ax.axvline(reference_value, color=COLOR_ACCENT, lw=2, ls="--",
                    label=f"{reference_label} = {reference_value:.4f}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title(f"Block-bootstrap distribution -- {metric_name}")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_training_curves(loss_histories: dict, save_path=None):
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for i, (seed, losses) in enumerate(loss_histories.items()):
        ax.plot(losses, color=PALETTE[i % len(PALETTE)], lw=1.6, label=f"seed {seed}")
    ax.set_xlabel("Logged step (every 5 epochs)")
    ax.set_ylabel("Hooke loss")
    ax.set_yscale("log")
    ax.set_title("Training loss by ensemble seed")
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all_figures(bundle: dict, out_dir) -> list:
    """`bundle` holds every array/scalar the plots above need. See
    `build_report()` in main_v304_soft_labels.py for exactly what it
    contains. Returns the list of saved file paths (skips any figure whose
    required keys are missing rather than raising)."""
    out_dir = Path(out_dir)
    paths = []

    paths.append(plot_roc_pr(bundle["y_true"], bundle["psi"], out_dir / "01_roc_pr.png"))
    paths.append(plot_calibration(bundle["y_true"], bundle["p_calib"],
                                   save_path=out_dir / "02_calibration.png"))
    paths.append(plot_confusion_matrices(bundle["cm_base"], bundle["cm_calib"],
                                          save_path=out_dir / "03_confusion.png"))
    paths.append(plot_psi_timeseries(bundle["dates"], bundle["psi_scaled"], bundle["y_true"],
                                      bundle.get("crisis_windows"), bundle.get("tau_scaled"),
                                      save_path=out_dir / "04_psi_timeseries.png"))
    paths.append(plot_psi_distribution(bundle["psi"], bundle["y_true"], bundle.get("cohens_d"),
                                        save_path=out_dir / "05_psi_distribution.png"))
    paths.append(plot_threshold_sweep(bundle["thresholds"], bundle["recalls"], bundle["precisions"],
                                       bundle["f1s"], bundle.get("tau_star"),
                                       save_path=out_dir / "06_threshold_sweep.png"))
    paths.append(plot_physics_correlations(bundle["gamma"], bundle["temp"], bundle["alpha"],
                                            bundle["psi"], save_path=out_dir / "07_physics_correlations.png"))

    if bundle.get("bootstrap_roc_auc") is not None:
        b = bundle["bootstrap_roc_auc"]
        paths.append(plot_bootstrap_distribution(
            b["samples"], b["point"], bundle.get("reference_roc_auc"),
            metric_name="ROC-AUC", save_path=out_dir / "08_bootstrap_roc_auc.png"))

    if bundle.get("loss_histories"):
        paths.append(plot_training_curves(bundle["loss_histories"],
                                           save_path=out_dir / "09_training_curves.png"))

    return [p for p in paths if p is not None]
