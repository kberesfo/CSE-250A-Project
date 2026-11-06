"""eval/concentration.py: bins → concentration, and its error."""

import numpy as np
import pandas as pd
import pytest

from cse250a.modules.eval.concentration import (
    bin_means,
    concentration_error,
    expected_concentration,
    resolution_floor,
)


@pytest.fixture
def train():
    raw = pd.Series([1.0, 2.0, 3.0, 10.0, 12.0, np.nan], name="CO")
    binned = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], name="CO")
    return raw, binned


def test_bin_means_per_bin_ignoring_missing(train):
    c_bar = bin_means(*train)
    assert c_bar.to_dict() == {0.0: 2.0, 1.0: 11.0}


def test_bin_medians(train):
    raw, binned = train
    assert bin_means(raw, binned, stat="median").to_dict() == {0.0: 2.0, 1.0: 11.0}


def test_bin_means_needs_the_same_rows(train):
    raw, binned = train
    with pytest.raises(ValueError, match="index"):
        bin_means(raw, binned.iloc[:-1])


def test_expected_concentration_is_probability_weighted_mean():
    proba = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]], index=["a", "b", "c"])
    c_bar = pd.Series({0: 2.0, 1: 10.0})
    est = expected_concentration(proba, c_bar)
    np.testing.assert_allclose(est, [2.0, 6.0, 8.0])
    assert est.index.tolist() == ["a", "b", "c"]


def test_expected_concentration_aligns_columns_by_label():
    proba = pd.DataFrame([[0.0, 1.0]], columns=[1, 0])  # columns out of order
    c_bar = pd.Series({0: 2.0, 1: 10.0})
    assert expected_concentration(proba, c_bar).iloc[0] == 2.0


def test_expected_concentration_missing_bin_mean():
    proba = pd.DataFrame([[0.5, 0.25, 0.25]])
    with pytest.raises(ValueError, match="bins \\[2\\]"):
        expected_concentration(proba, pd.Series({0: 1.0, 1: 2.0}))


def test_concentration_error_skips_missing_truth():
    est = pd.Series([1.0, 2.0, 3.0])
    truth = pd.Series([2.0, np.nan, 1.0])
    r = concentration_error(est, truth)
    assert r["n"] == 2
    assert r["MAE"] == pytest.approx(1.5)            # |−1|, |2|
    assert r["RMSE"] == pytest.approx(np.sqrt(2.5))  # sqrt((1 + 4) / 2)


def test_resolution_floor_is_the_error_of_a_perfect_bin_prediction(train):
    raw, binned = train
    floor = resolution_floor(raw, binned, bin_means(raw, binned))
    assert floor["n"] == 5
    assert floor["MAE"] == pytest.approx((1 + 0 + 1 + 1 + 1) / 5)
