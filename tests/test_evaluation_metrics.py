"""Tests for evaluation_metrics.py. Pure numpy/sklearn, no torch."""
import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

import evaluation_metrics as em


# ---------------------------------------------------------------------------
# Point-estimate metrics
# ---------------------------------------------------------------------------

def test_cohens_d_is_large_for_well_separated_groups():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 200 + [1] * 200)
    scores = np.concatenate([rng.normal(0.2, 0.05, 200), rng.normal(0.8, 0.05, 200)])
    d = em.cohens_d(scores, y_true)
    assert d > 3.0  # groups 0.6 apart with std 0.05 -> a very large effect


def test_cohens_d_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    y_true = np.array([0] * 500 + [1] * 500)
    scores = rng.normal(0.5, 0.1, 1000)
    d = em.cohens_d(scores, y_true)
    assert abs(d) < 0.3


def test_youdens_j_is_one_for_a_perfect_classifier():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    assert em.youdens_j(y_true, y_pred) == pytest.approx(1.0)


def test_youdens_j_is_zero_for_always_predict_positive():
    # sensitivity=1, specificity=0 -> J = 1 + 0 - 1 = 0
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    assert em.youdens_j(y_true, y_pred) == pytest.approx(0.0, abs=1e-6)


def test_brier_skill_score_is_zero_when_predicting_the_base_rate():
    y_true = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0])
    base_rate = y_true.mean()
    y_prob = np.full_like(y_true, base_rate, dtype=float)
    assert em.brier_skill_score(y_true, y_prob) == pytest.approx(0.0, abs=1e-6)


def test_brier_skill_score_is_positive_for_a_better_than_baseline_model():
    rng = np.random.default_rng(2)
    y_true = np.array([0] * 800 + [1] * 200)
    y_prob = np.concatenate([rng.uniform(0.0, 0.3, 800), rng.uniform(0.7, 1.0, 200)])
    assert em.brier_skill_score(y_true, y_prob) > 0.5


def test_calibration_slope_near_one_for_genuinely_calibrated_probabilities():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.99, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    slope, intercept = em.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_expected_calibration_error_is_near_zero_for_calibrated_probabilities():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.01, 0.99, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    ece = em.expected_calibration_error(y, p)
    assert ece < 0.05


def test_expected_calibration_error_is_large_for_badly_miscalibrated_probabilities():
    y = np.array([0] * 100)  # always stable...
    p = np.full(100, 0.9)  # ...but the model is 90% sure it's a crisis every time
    ece = em.expected_calibration_error(y, p)
    assert ece == pytest.approx(0.9, abs=1e-6)


def test_legacy_average_precision_matches_sklearn_without_ties():
    rng = np.random.default_rng(5)
    y_true = np.array([0] * 800 + [1] * 200)
    scores = np.concatenate([rng.beta(2, 5, 800), rng.beta(5, 2, 200)])
    legacy = em.legacy_average_precision(y_true, scores)
    sklearn_value = average_precision_score(y_true, scores)
    assert legacy == pytest.approx(sklearn_value, abs=1e-6)


def test_point_estimates_returns_all_expected_keys():
    rng = np.random.default_rng(6)
    n = 300
    y_true = (rng.uniform(0, 1, n) < 0.15).astype(int)
    scores = np.clip(rng.beta(2, 5, n) + y_true * 0.3, 0, 1)
    y_prob = scores
    y_pred = (scores >= 0.5).astype(int)
    result = em.point_estimates(y_true, scores, y_prob, y_pred)
    expected_keys = {
        "roc_auc", "pr_auc", "cohens_d", "mcc", "balanced_accuracy",
        "youdens_j", "brier_skill_score", "calibration_slope", "calibration_intercept",
    }
    assert expected_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# Block bootstrap
# ---------------------------------------------------------------------------

def test_block_bootstrap_ci_contains_the_point_estimate():
    rng = np.random.default_rng(7)
    n = 500
    y_true = (rng.uniform(0, 1, n) < 0.2).astype(int)
    scores = np.clip(rng.beta(2, 4, n) + y_true * 0.35, 0, 1)
    result = em.block_bootstrap_ci(y_true, scores, roc_auc_score, n_boot=300, block_size=21, seed=1)
    assert result["lo"] <= result["point"] <= result["hi"]
    assert result["n_valid_boot"] > 0


def test_block_bootstrap_ci_is_reproducible_with_the_same_seed():
    rng = np.random.default_rng(8)
    n = 400
    y_true = (rng.uniform(0, 1, n) < 0.2).astype(int)
    scores = np.clip(rng.beta(2, 4, n) + y_true * 0.35, 0, 1)
    r1 = em.block_bootstrap_ci(y_true, scores, roc_auc_score, n_boot=200, block_size=21, seed=42)
    r2 = em.block_bootstrap_ci(y_true, scores, roc_auc_score, n_boot=200, block_size=21, seed=42)
    assert r1["lo"] == pytest.approx(r2["lo"])
    assert r1["hi"] == pytest.approx(r2["hi"])


def test_reference_percentile_extremes():
    samples = np.array([0.80, 0.82, 0.85, 0.88, 0.90])
    assert em.reference_percentile(samples, 0.0) == pytest.approx(0.0)
    assert em.reference_percentile(samples, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

def test_run_history_round_trips_through_disk(tmp_path):
    path = tmp_path / "run_history.json"
    em.save_run_record(path, "V1", {"roc_auc": 0.80})
    em.save_run_record(path, "V2", {"roc_auc": 0.85})
    history = em.load_run_history(path)
    assert history == {"V1": {"roc_auc": 0.80}, "V2": {"roc_auc": 0.85}}
    # File on disk should be valid, readable JSON independent of the loader.
    assert json.loads(path.read_text())["V2"]["roc_auc"] == 0.85


def test_load_run_history_returns_empty_dict_when_file_missing(tmp_path):
    assert em.load_run_history(tmp_path / "does_not_exist.json") == {}


def test_compare_versions_flags_the_correct_winner_per_direction():
    history = {
        "V1": {"roc_auc": 0.80, "ece": 5.0},
        "V2": {"roc_auc": 0.85, "ece": 6.0},
    }
    rows = em.compare_versions(history, "V2", "V1", lower_is_better=("ece",))
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["roc_auc"]["current_is_better"] is True  # higher roc_auc, higher is better
    assert by_metric["ece"]["current_is_better"] is False  # higher ece, but lower is better


def test_compare_versions_returns_empty_when_a_version_is_missing():
    history = {"V1": {"roc_auc": 0.80}}
    assert em.compare_versions(history, "V2", "V1") == []
