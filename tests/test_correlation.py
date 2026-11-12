"""eval/correlation.py: Spearman matrices and partial Spearman."""

import numpy as np
import pandas as pd
import pytest

from cse250a.modules.eval.correlation import partial_spearman, spearman_matrix


@pytest.fixture
def traffic():
    """y depends on a shared driver z only; x depends on z and its own signal."""
    rng = np.random.default_rng(0)
    n = 4000
    z = rng.normal(size=n)
    own = rng.normal(size=n)
    return pd.DataFrame(
        {
            "z": z,
            "x": z + own,
            "y": z + 0.3 * rng.normal(size=n),
            "w": own + 0.3 * rng.normal(size=n),
            "hour": rng.integers(0, 24, size=n),
        }
    )


def test_no_controls_matches_pandas_spearman(traffic):
    rho, n = partial_spearman(traffic, "x", "y")
    assert n == len(traffic)
    assert rho == pytest.approx(traffic["x"].corr(traffic["y"], method="spearman"))


def test_shared_driver_is_removed(traffic):
    raw, _ = partial_spearman(traffic, "x", "y")
    partial, _ = partial_spearman(traffic, "x", "y", ["z"])
    assert raw > 0.5
    assert abs(partial) < 0.1  # x and y only share z


def test_own_signal_survives_control(traffic):
    partial, _ = partial_spearman(traffic, "x", "w", ["z"])
    assert partial > 0.5


def test_categorical_control_and_missing_rows(traffic):
    df = traffic.copy()
    df.loc[:9, "y"] = np.nan
    rho, n = partial_spearman(df, "x", "y", ["z", "hour"], categorical=["hour"])
    assert n == len(df) - 10
    assert np.isfinite(rho)


def test_spearman_matrix_counts_pairs(traffic):
    df = traffic.copy()
    df.loc[:99, "w"] = np.nan
    rho, n = spearman_matrix(df, ["x", "y"], ["x", "w"])
    assert rho.loc["x", "x"] == 1.0
    assert n.loc["y", "w"] == len(df) - 100
    assert n.loc["y", "x"] == len(df)
    assert rho.loc["x", "w"] == pytest.approx(df["x"].corr(df["w"], method="spearman"))
