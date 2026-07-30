"""Smoke tests for visualization.py: every figure function should run
without raising and produce a non-empty PNG. These do not check pixel
content -- only that the plotting code paths are exercised end to end on
data shaped like the real pipeline's outputs."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import visualization as viz


@pytest.fixture
def synthetic_report_data():
    rng = np.random.default_rng(0)
    n = 1041
    dates = pd.bdate_range("2020-01-01", periods=n)
    y_true = np.zeros(n, dtype=int)
    y_true[100:150] = 1
    y_true[400:470] = 1
    psi = np.clip(rng.beta(2, 5, n) + y_true * rng.uniform(0.3, 0.6, n), 0, 1)
    p_calib = np.clip(psi + rng.normal(0, 0.05, n), 1e-4, 1 - 1e-4)
    gamma = rng.uniform(0, 1, n)
    temp = np.clip(0.05 + 0.3 * psi + rng.normal(0, 0.02, n), 0, 1)
    alpha = rng.uniform(0.4, 0.95, n)
    return dict(dates=dates, y_true=y_true, psi=psi, p_calib=p_calib,
                gamma=gamma, temp=temp, alpha=alpha)


def _assert_saved_png(path):
    p = Path(path)
    assert p.exists()
    assert p.stat().st_size > 1000  # a real rendered figure, not an empty file


def test_plot_roc_pr(tmp_path, synthetic_report_data):
    d = synthetic_report_data
    out = viz.plot_roc_pr(d["y_true"], d["psi"], save_path=tmp_path / "roc_pr.png")
    _assert_saved_png(out)


def test_plot_calibration(tmp_path, synthetic_report_data):
    d = synthetic_report_data
    out = viz.plot_calibration(d["y_true"], d["p_calib"], save_path=tmp_path / "cal.png")
    _assert_saved_png(out)


def test_plot_confusion_matrices(tmp_path):
    cm_base = np.array([[863, 18], [71, 89]])
    cm_calib = np.array([[829, 52], [42, 118]])
    out = viz.plot_confusion_matrices(cm_base, cm_calib, save_path=tmp_path / "cm.png")
    _assert_saved_png(out)


def test_plot_psi_timeseries(tmp_path, synthetic_report_data):
    d = synthetic_report_data
    out = viz.plot_psi_timeseries(
        d["dates"], d["psi"] * 100, d["y_true"],
        crisis_windows=[("Demo crisis", "2020-04-10", "2020-06-01")],
        threshold=25.0, save_path=tmp_path / "psi_ts.png",
    )
    _assert_saved_png(out)


def test_plot_psi_distribution(tmp_path, synthetic_report_data):
    d = synthetic_report_data
    out = viz.plot_psi_distribution(d["psi"], d["y_true"], cohens_d_value=1.5,
                                     save_path=tmp_path / "psi_dist.png")
    _assert_saved_png(out)


def test_plot_threshold_sweep(tmp_path):
    thresholds = [0.15, 0.25, 0.40, 0.65]
    recalls = [0.79, 0.74, 0.68, 0.62]
    precisions = [0.65, 0.69, 0.76, 0.81]
    f1s = [0.71, 0.72, 0.71, 0.70]
    out = viz.plot_threshold_sweep(thresholds, recalls, precisions, f1s, tau_star=0.25,
                                    save_path=tmp_path / "sweep.png")
    _assert_saved_png(out)


def test_plot_physics_correlations(tmp_path, synthetic_report_data):
    d = synthetic_report_data
    out = viz.plot_physics_correlations(d["gamma"], d["temp"], d["alpha"], d["psi"],
                                         save_path=tmp_path / "physics.png")
    _assert_saved_png(out)


def test_plot_bootstrap_distribution(tmp_path):
    rng = np.random.default_rng(9)
    samples = rng.normal(0.91, 0.02, 2000)
    out = viz.plot_bootstrap_distribution(samples, point_estimate=0.91, reference_value=0.9088,
                                           metric_name="ROC-AUC", save_path=tmp_path / "boot.png")
    _assert_saved_png(out)


def test_plot_training_curves(tmp_path):
    loss_histories = {
        42: list(0.5 * np.exp(-0.1 * np.arange(30)) + 0.02),
        123: list(0.6 * np.exp(-0.08 * np.arange(30)) + 0.025),
    }
    out = viz.plot_training_curves(loss_histories, save_path=tmp_path / "curves.png")
    _assert_saved_png(out)


def test_generate_all_figures_writes_every_figure(tmp_path, synthetic_report_data):
    d = synthetic_report_data
    p65 = (d["psi"] >= 0.65).astype(int)
    p25 = (d["p_calib"] >= 0.25).astype(int)

    def _cm(pred):
        tn = int(((d["y_true"] == 0) & (pred == 0)).sum())
        fp = int(((d["y_true"] == 0) & (pred == 1)).sum())
        fn = int(((d["y_true"] == 1) & (pred == 0)).sum())
        tp = int(((d["y_true"] == 1) & (pred == 1)).sum())
        return np.array([[tn, fp], [fn, tp]])

    bundle = dict(
        y_true=d["y_true"], psi=d["psi"], p_calib=d["p_calib"],
        cm_base=_cm(p65), cm_calib=_cm(p25),
        dates=d["dates"], psi_scaled=d["psi"] * 100, tau_scaled=25.0,
        crisis_windows=[("Demo crisis", "2020-04-10", "2020-06-01")], cohens_d=1.5,
        thresholds=[0.15, 0.25, 0.40], recalls=[0.79, 0.74, 0.68],
        precisions=[0.65, 0.69, 0.76], f1s=[0.71, 0.72, 0.71], tau_star=0.25,
        gamma=d["gamma"], temp=d["temp"], alpha=d["alpha"],
        bootstrap_roc_auc=None, reference_roc_auc=None, loss_histories=None,
    )
    paths = viz.generate_all_figures(bundle, tmp_path)
    assert len(paths) == 7  # the two optional figures (bootstrap, training curves) are skipped here
    for p in paths:
        _assert_saved_png(p)
