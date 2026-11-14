"""eval/latent.py and the conditional-correlation helpers in eval/correlation.py."""

import numpy as np
import pandas as pd
import pytest

from cse250a.modules.core import CPT, Network
from cse250a.modules.eval import (
    free_params,
    restart_agreement,
    state_order,
    state_profile,
    weighted_corr,
    within_group_spearman,
    within_state_spearman,
)


class _BN:
    """Minimal stand-in: a network plus CPT tables, as LayeredBN/DiscreteBN expose them."""

    def __init__(self, network, tables):
        self.network = network
        self.cpts = {
            n: CPT(n, network.structure[n], np.asarray(t)) for n, t in tables.items()
        }


def _event_bn(order):
    """K = 3 latent 'Event' → two binary-ish children with 3 bins; states listed in `order`."""
    co = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]])[order]
    no2 = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])[order]
    net = Network(
        structure={"Event": ("hour",), "CO(GT)": ("Event",), "NO2(GT)": ("Event",)},
        cards={"Event": 3, "hour": 24, "CO(GT)": 3, "NO2(GT)": 3},
        latent=("Event",),
    )
    event = np.full((24, 3), 1 / 3)
    return _BN(net, {"Event": event, "CO(GT)": co, "NO2(GT)": no2})


def test_free_params_counts_each_cpt():
    bn = _event_bn([0, 1, 2])
    # Event: 24 × 2, each gas: 3 × 2
    assert free_params(bn.network) == 24 * 2 + 2 * (3 * 2)


def test_state_profile_and_order():
    bn = _event_bn([2, 0, 1])  # states shuffled: high, low, medium
    prof = state_profile(bn, "Event", ["CO(GT)", "NO2(GT)"])
    assert prof.shape == (3, 6)
    assert list(state_order(bn, "Event", "CO(GT)")) == [1, 2, 0]


def test_state_profile_requires_latent_first_parent():
    bn = _event_bn([0, 1, 2])
    with pytest.raises(ValueError):
        state_profile(bn, "hour", ["CO(GT)"])


def test_restart_agreement_ignores_relabeling():
    a = state_profile(_event_bn([0, 1, 2]), "Event", ["CO(GT)", "NO2(GT)"])
    b = state_profile(_event_bn([2, 0, 1]), "Event", ["CO(GT)", "NO2(GT)"])
    c = b.copy()
    c[0, :3] = [1 / 3, 1 / 3, 1 / 3]  # a genuinely different state
    d_same, d_diff = restart_agreement([a, b, c])
    assert d_same == pytest.approx(0.0)
    assert d_diff > 0.1


@pytest.fixture
def two_regimes():
    """x and y move together only through a hidden regime; within it they're independent."""
    rng = np.random.default_rng(0)
    n = 6000
    state = rng.integers(0, 2, size=n)
    x = 3 * state + rng.normal(size=n)
    y = 3 * state + rng.normal(size=n)
    hour = rng.integers(0, 24, size=n)
    return pd.DataFrame({"x": x, "y": y, "hour": hour}), np.eye(2)[state]


def test_weighted_corr_matches_pearson_with_equal_weights(two_regimes):
    df, _ = two_regimes
    w = np.ones(len(df))
    assert weighted_corr(df["x"].to_numpy(), df["y"].to_numpy(), w) == pytest.approx(
        df["x"].corr(df["y"])
    )


def test_within_state_removes_regime_driven_correlation(two_regimes):
    df, post = two_regimes
    raw = df["x"].corr(df["y"], method="spearman")
    within_hour = within_group_spearman(df, "x", "y", "hour")
    within_state = within_state_spearman(df, "x", "y", post)
    assert raw > 0.5
    assert within_hour > 0.5  # hour is unrelated to the regime, so it explains nothing
    assert abs(within_state) < 0.2  # ranks are global, so a little dependence remains


def test_within_group_skips_constant_groups(two_regimes):
    df, _ = two_regimes
    df = df.copy()
    df.loc[df["hour"] == 0, "y"] = 1.0
    assert np.isfinite(within_group_spearman(df, "x", "y", "hour"))
