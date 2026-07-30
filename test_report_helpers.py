"""Tests for _confusion_and_rates -- the tp/tn/fp/fn arithmetic shared by
every table in build_report(). Pure numpy, no torch involved."""
import numpy as np
import pytest

import main_v304_soft_labels as mvsl


def test_confusion_and_rates_on_a_hand_computed_example():
    # 10 samples: 4 true crises (indices 0-3), 6 stable (indices 4-9).
    # Predictions: catches 3 of 4 crises (1 false negative), raises 2 false
    # alarms on stable days.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])

    rates = mvsl._confusion_and_rates(y_true, y_pred)

    assert rates["tp"] == 3
    assert rates["fn"] == 1
    assert rates["fp"] == 2
    assert rates["tn"] == 4
    assert rates["precision"] == pytest.approx(3 / 5, rel=1e-6)
    assert rates["recall"] == pytest.approx(3 / 4, rel=1e-6)
    assert rates["accuracy"] == pytest.approx(7 / 10, rel=1e-6)
    assert rates["fpr"] == pytest.approx(2 / 6, rel=1e-6)
    expected_f1 = 2 * rates["precision"] * rates["recall"] / (rates["precision"] + rates["recall"])
    assert rates["f1"] == pytest.approx(expected_f1, rel=1e-6)


def test_confusion_and_rates_perfect_classifier():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    rates = mvsl._confusion_and_rates(y_true, y_pred)
    assert rates["precision"] == pytest.approx(1.0, rel=1e-6)
    assert rates["recall"] == pytest.approx(1.0, rel=1e-6)
    assert rates["f1"] == pytest.approx(1.0, rel=1e-6)
    assert rates["fpr"] == pytest.approx(0.0, abs=1e-6)


def test_confusion_and_rates_confusion_matrix_field_matches_counts():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    rates = mvsl._confusion_and_rates(y_true, y_pred)
    tn, fp, fn, tp = rates["tn"], rates["fp"], rates["fn"], rates["tp"]
    np.testing.assert_array_equal(rates["confusion_matrix"], np.array([[tn, fp], [fn, tp]]))
