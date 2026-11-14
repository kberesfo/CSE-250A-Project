"""models/layered.py: LayeredBN must match DiscreteBN exactly, just faster.

Checked on a slice of the real data (so missing gases and sensors are included)
for the three structures it is used for: Phase 3b Event-only, Event + hour, and
Phase 2 (no latent node).
"""

import numpy as np
import pytest

from cse250a.modules.config import config
from cse250a.modules.core import TIME_CARDS, Gas, Network, Sensor, Time, Weather
from cse250a.modules.data import Discretizer, blocked_split, load_air_quality
from cse250a.modules.models import DiscreteBN
from cse250a.modules.models.layered import LayeredBN

W = (Weather.T, Weather.AH)
GASES = (Gas.CO, Gas.C6H6, Gas.NOX, Gas.NO2)
NOMINAL = {Sensor.S1: (Gas.CO,), Sensor.S3: (Gas.NOX,), Sensor.S4: (Gas.NO2,)}
CROSS = {
    Sensor.S2: (Gas.C6H6,),
    Sensor.S4: (Gas.CO, Gas.NOX),
    Sensor.S5: (Gas.CO, Gas.NOX),
}
SENSOR_PARENTS = {s: (*NOMINAL.get(s, ()), *CROSS.get(s, ()), *W) for s in Sensor}


@pytest.fixture(scope="module")
def data():
    df = load_air_quality(config.paths.air_quality_csv).dropna(subset=[Time.HOUR, *W])
    train_raw, val_raw, _ = blocked_split(df, seed=0)
    disc = Discretizer(n_bins=3).fit(train_raw, [*Sensor, *GASES, *W])
    train, val = disc.transform(train_raw), disc.transform(val_raw)
    # a slice with plenty of missing cells, small enough for DiscreteBN
    miss = train[list(GASES)].isna().any(axis=1) | train[list(Sensor)].isna().any(
        axis=1
    )
    rows = train[miss].head(60).index.union(train[~miss].head(140).index)
    return train.loc[rows], val.head(150), {**disc.cards, **TIME_CARDS}


def networks(cards):
    event = {"Event": (Time.HOUR, Time.IS_WEEKDAY)}
    yield (
        "event_only",
        Network(
            structure={**event, **{g: ("Event",) for g in GASES}, **SENSOR_PARENTS},
            cards={**cards, "Event": 3},
            latent=("Event",),
        ),
    )
    yield (
        "event_hour",
        Network(
            structure={
                "Event": (Time.HOUR, Time.DOW),
                **{g: ("Event", Time.HOUR) for g in GASES},
                **SENSOR_PARENTS,
            },
            cards={**cards, "Event": 2},
            latent=("Event",),
        ),
    )
    yield (
        "phase2",
        Network(
            structure={**{g: (Time.HOUR,) for g in GASES}, **SENSOR_PARENTS},
            cards=cards,
        ),
    )


@pytest.mark.parametrize("which", ["event_only", "event_hour", "phase2"])
def test_matches_discrete_bn(data, which):
    train, val, cards = data
    net = dict(networks(cards))[which]
    slow = DiscreteBN(net, seed=3, max_iter=4, tol=0).fit(train)
    fast = LayeredBN(net, seed=3, max_iter=4, tol=0).fit(train)

    # same EM path: same objective at every iteration
    np.testing.assert_allclose(fast.result.history, slow.result.history, rtol=1e-9)
    for node in net.structure:
        np.testing.assert_allclose(
            fast.cpts[node].table, slow.cpts[node].table, atol=1e-10
        )

    # E-step on held-out rows, from the same parameters
    c_fast, ll_fast = fast._expected_counts(val)
    c_slow, ll_slow = slow._expected_counts(val)
    assert ll_fast == pytest.approx(ll_slow, rel=1e-10)
    for node in net.structure:
        np.testing.assert_allclose(c_fast[node], c_slow[node], atol=1e-8)

    assert fast.log_likelihood(val) == pytest.approx(
        slow.log_likelihood(val), rel=1e-10
    )
    for node in (*net.latent, Gas.CO, Sensor.S4):
        np.testing.assert_allclose(
            fast.posterior(val, node), slow.posterior(val, node), atol=1e-10
        )


def test_predict_proba_masks_like_discrete_bn(data):
    train, val, cards = data
    net = dict(networks(cards))["event_only"]
    slow = DiscreteBN(net, seed=1, max_iter=3).fit(train)
    fast = LayeredBN(net, seed=1, max_iter=3).fit(train)
    evidence = (Time.HOUR, Time.IS_WEEKDAY, *W, *Sensor)
    np.testing.assert_allclose(
        fast.predict_proba(val, Gas.NO2, evidence).to_numpy(),
        slow.predict_proba(val, Gas.NO2, evidence).to_numpy(),
        atol=1e-10,
    )


def test_rejects_unsupported_structure(data):
    _, _, cards = data
    chain = Network(
        structure={Gas.CO: (Time.HOUR,), Gas.NO2: (Gas.CO,), Sensor.S4: (Gas.NO2,)},
        cards=cards,
    )
    with pytest.raises(ValueError):
        LayeredBN(chain)
