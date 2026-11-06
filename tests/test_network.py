"""core/network.py: the Network container and its validation.

Which network a phase uses is decided in its notebook, so nothing here tests
particular edges; only that any network is read and checked correctly.
"""

import pickle

import pytest

from cse250a.modules.core import COLUMNS, Gas, Network, Sensor, Time, Weather

H, W = Time.HOUR, Weather.T
A, B, C = Gas.CO, Sensor.S1, Sensor.S4


def tiny_network(**overrides):
    kwargs = {
        "structure": {B: (A, W), A: (H,), C: (A,)},  # deliberately not in order
        "cards": {H: 24, W: 3, A: 3, B: 2, C: 2},
    }
    return Network(**(kwargs | overrides))


# ── reading the graph ──────────────────────────────────────────────────────────


def test_nodes_are_topologically_ordered():
    nodes = tiny_network().nodes
    assert set(nodes) == {A, B, C}
    assert nodes.index(A) < nodes.index(B) and nodes.index(A) < nodes.index(C)


def test_diamond_is_not_a_cycle():
    d = Gas.NO2
    net = Network({A: (), B: (A,), C: (A,), d: (B, C)}, {A: 2, B: 2, C: 2, d: 2})
    assert net.nodes[0] == A and net.nodes[-1] == d


def test_inputs_in_first_seen_order():
    assert tiny_network().inputs == (W, H)  # B's parents come first in structure


def test_observed_and_children():
    z = Gas.CO.latent
    net = Network({z: (H,), B: (z, W), C: (z,)}, {H: 24, W: 3, z: 3, B: 2, C: 2}, (z,))
    assert set(net.observed) == {B, C}
    assert set(net.children(z)) == {B, C}
    assert net.children(B) == ()


def test_family_and_shape_are_parents_then_node():
    net = tiny_network()
    assert net.family(B) == (A, W, B)
    assert net.shape(B) == (3, 3, 2)
    assert net.shape(A) == (24, 3)


def test_temporal_shape_puts_previous_hour_first():
    net = tiny_network(prev={A: (A,)})
    assert net.is_temporal
    assert net.shape(A) == (3, 24, 3)  # A_{t-1}, H, A
    assert not tiny_network().is_temporal
    assert not tiny_network(prev={A: ()}).is_temporal, "empty prev parents → static"


# ── immutability and saving ────────────────────────────────────────────────────


def test_copies_the_callers_dicts():
    structure = {A: [H], B: [A, W], C: [A]}  # lists on purpose
    net = tiny_network(structure=structure)
    assert net.structure[B] == (A, W), "parents stored as tuples"
    structure[A].append(W)
    assert net.structure[A] == (H,), "caller's edit must not reach the network"


def test_is_frozen():
    with pytest.raises(AttributeError):
        tiny_network().structure = {}


def test_pickles():
    net = tiny_network()
    assert pickle.loads(pickle.dumps(net)) == net


# ── validate ───────────────────────────────────────────────────────────────────


def test_validate_returns_self_when_fine():
    net = tiny_network()
    assert net.validate() is net
    assert net.validate(columns=[H, W, A, B, C]) is net


def test_plain_strings_work_like_enums():
    net = Network({"CO(GT)": ("hour",)}, {"CO(GT)": 3, "hour": 24}).validate()
    assert net.inputs == (H,)


@pytest.mark.parametrize(
    "overrides, names",
    [
        ({"structure": {A: (H, B), B: (A, W), C: (A,)}}, r"cycle"),  # A ↔ B
        ({"cards": {H: 24, W: 3, A: 3, B: 2}}, r"cards.*PT08\.S4"),  # C has no card
        ({"cards": {H: 24, W: 3, A: 3, B: 2, C: 0}}, r"≥ 1"),  # a card of 0
        ({"structure": {A: ("hr",)}, "cards": {A: 3, "hr": 24}}, r"hr"),  # typo
        ({"latent": ("Gas_NO2",)}, r"Gas_NO2"),  # latent not in structure
        ({"latent": (A,)}, r"latent.*CO\(GT\)"),  # latent named like a real column
        ({"prev": {A: ("ghost",)}}, r"ghost"),  # prev names an unknown node
    ],
)
def test_validate_reports_problems(overrides, names):
    """Each case has one problem; the message should name the culprit."""
    with pytest.raises(ValueError, match=names):
        tiny_network(**overrides).validate()


def test_forgetting_latent_is_reported_as_unknown_name():
    """A latent node not declared in `latent` must not silently become hidden."""
    z = Gas.CO.latent
    with pytest.raises(ValueError, match=r"Gas_CO"):
        Network({z: (H,), B: (z,)}, {H: 24, z: 3, B: 2}).validate()


def test_validate_collects_every_problem():
    with pytest.raises(ValueError) as e:
        Network({A: ("hr",)}, {}).validate()
    msg = str(e.value)
    assert "hr" in msg and "cards" in msg, "both problems in one error"


def test_validate_checks_columns():
    with pytest.raises(ValueError, match=r"PT08\.S4"):
        tiny_network().validate(columns=[H, W, A, B])


def test_latent_is_not_expected_as_a_column():
    z = Gas.CO.latent
    net = Network({z: (H,), B: (z,)}, {H: 24, z: 3, B: 2}, (z,))
    net.validate(columns=[H, B])  # no Gas_CO column, and that's fine


def test_columns_vocabulary():
    assert {Gas.CO, Sensor.S1, Weather.T, Time.HOUR} <= COLUMNS
    assert Gas.CO.latent not in COLUMNS


def test_latent_nodes_can_be_parents_or_children_of_gases():
    """Phase 3a: AQ is a child of every gas. Phase 3b: Event is a parent of them.
    Any name works for a latent node, as long as it's declared and isn't a column."""
    no2 = Gas.NO2
    net = Network(
        {"Event": (H,), A: ("Event",), no2: ("Event",), "AQ": (A, no2), B: (A,)},
        {H: 24, "Event": 2, A: 3, no2: 3, "AQ": 3, B: 2},
        latent=("Event", "AQ"),
    ).validate(columns=[H, A, no2, B])
    assert set(net.children("Event")) == {A, no2}
    assert net.shape("AQ") == (3, 3, 3)
