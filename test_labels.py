"""Tests for make_hard_labels / make_soft_labels.

These only exercise pandas/numpy/scipy code paths -- no torch model is
instantiated, so these tests run fast and need no GPU.
"""
import numpy as np
import pandas as pd
import pytest

import main_v304_soft_labels as mvsl


@pytest.fixture(scope="module")
def date_index():
    # Matches the project's real historical span (config.py START_DATE /
    # END_DATE) so tests can reference any CRISES window, not just a slice.
    return pd.bdate_range("1993-02-01", "2023-12-31")


def test_hard_label_is_one_inside_a_known_crisis_window(date_index):
    hard = mvsl.make_hard_labels(date_index)
    # 2020-03-15 falls inside the COVID window (2020-02-20 to 2020-04-30).
    assert hard.loc["2020-03-16"] == 1


def test_hard_label_is_zero_on_an_ordinary_day(date_index):
    hard = mvsl.make_hard_labels(date_index)
    # 2019-06-17 is not inside any CRISES window.
    assert hard.loc["2019-06-17"] == 0


def test_hard_label_covers_every_named_crisis_boundary(date_index):
    hard = mvsl.make_hard_labels(date_index)
    for start, end in mvsl.CRISES:
        if pd.Timestamp(start) < date_index.min() or pd.Timestamp(end) > date_index.max():
            continue
        window = hard.loc[start:end]
        assert (window == 1).all(), f"window {start}..{end} is not fully flagged"


def test_soft_labels_are_bounded_in_unit_interval(date_index):
    soft = mvsl.make_soft_labels(date_index, sigma=7)
    assert soft.min() >= 0.0
    assert soft.max() <= 1.0


def test_soft_labels_match_hard_labels_far_from_any_crisis_edge(date_index):
    hard = mvsl.make_hard_labels(date_index)
    soft = mvsl.make_soft_labels(date_index, sigma=7)
    # Deep inside the 2008-09-01..2009-03-31 window (well over 5*sigma from
    # either edge), the Gaussian ramp should have fully saturated.
    assert soft.loc["2008-12-01"] == pytest.approx(1.0, abs=1e-3)
    assert hard.loc["2008-12-01"] == 1
    # Deep inside a long stable stretch, equally saturated toward 0.
    assert soft.loc["2013-06-03"] == pytest.approx(0.0, abs=1e-3)
    assert hard.loc["2013-06-03"] == 0


def test_soft_label_ramp_is_roughly_centered_at_crisis_onset(date_index):
    soft = mvsl.make_soft_labels(date_index, sigma=7)
    # COVID crisis starts 2020-02-20 and the window is long enough on both
    # sides for the ramp to be uncontaminated by the opposite edge -- the
    # smoothed value right at onset should sit close to the 0.5 midpoint.
    onset_value = soft.loc["2020-02-20"]
    assert 0.3 < onset_value < 0.7


def test_soft_label_defaults_to_cfg_sigma(date_index):
    default = mvsl.make_soft_labels(date_index)
    explicit = mvsl.make_soft_labels(date_index, sigma=mvsl.CFG.SOFT_LABEL_SIGMA)
    assert np.allclose(default.values, explicit.values)
