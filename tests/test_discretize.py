"""data/discretize.py: quantile and fixed-edge binning."""

import numpy as np
import pandas as pd
import pytest

from cse250a.modules.data import Discretizer


@pytest.fixture
def train():
    """x: 1..12 with one NaN (no ties); hour passes through."""
    return pd.DataFrame(
        {
            "x": [*range(1, 13), np.nan],
            "y": np.linspace(0.0, 100.0, 13),
            "hour": [*range(12), 0],
        },
        dtype=float,
    )


def test_quantile_edges_ignore_nan(train):
    disc = Discretizer(3).fit(train, ["x"])
    np.testing.assert_allclose(
        disc.edges["x"], np.nanquantile(train["x"], [1 / 3, 2 / 3])
    )
    assert len(disc.edges["x"]) == 2, "3 bins → 2 interior edges"


def test_transform_gives_equal_sized_bins(train):
    out = Discretizer(3).fit_transform(train, ["x"])
    bins = out["x"].dropna()
    assert set(bins.unique()) == {0.0, 1.0, 2.0}
    assert bins.value_counts().tolist() == [4, 4, 4], "quantile bins hold equal counts"
    np.testing.assert_array_equal(bins.iloc[:4], [0, 0, 0, 0])
    np.testing.assert_array_equal(bins.iloc[-4:], [2, 2, 2, 2])


def test_nan_stays_nan_and_other_columns_pass_through(train):
    out = Discretizer(3).fit_transform(train, ["x"])
    assert np.isnan(out["x"].iloc[-1])
    pd.testing.assert_series_equal(out["hour"], train["hour"])
    pd.testing.assert_series_equal(out["y"], train["y"])  # not in cols


def test_transform_does_not_change_its_input(train):
    before = train.copy()
    Discretizer(3).fit_transform(train, ["x"])
    pd.testing.assert_frame_equal(train, before)


def test_new_data_uses_training_edges(train):
    """Edges come from train only; out-of-range values clip into the end bins."""
    disc = Discretizer(3).fit(train, ["x"])
    new = pd.DataFrame({"x": [-50.0, 6.0, 500.0]})
    np.testing.assert_array_equal(disc.transform(new)["x"], [0, 1, 2])


def test_per_column_bins_and_cards(train):
    disc = Discretizer(3, per_column={"y": 5}).fit(train, ["x", "y"])
    assert disc.cards == {"x": 3, "y": 5}
    assert len(disc.edges["y"]) == 4
    assert disc.transform(train)["y"].max() == 4


def test_fixed_edges_are_used_as_given(train):
    disc = Discretizer(3, fixed_edges={"x": [2.5, 10.0]}).fit(train, ["x"])
    np.testing.assert_allclose(disc.edges["x"], [2.5, 10.0])
    assert disc.cards == {"x": 3}
    out = disc.transform(pd.DataFrame({"x": [1.0, 5.0, 11.0]}))
    np.testing.assert_array_equal(out["x"], [0, 1, 2])


def test_fixed_edges_must_increase(train):
    with pytest.raises(ValueError):
        Discretizer(fixed_edges={"x": [10.0, 2.5]}).fit(train, ["x"])
