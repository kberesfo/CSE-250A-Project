"""core/: columns, CPT, Network and the network builders, FitResult, Messages."""

import itertools

import numpy as np
import pandas as pd
import pytest

from cse250a.modules.core import (
    CPT,
    GASES,
    HOUR,
    SENSORS,
    WEATHER,
    FitResult,
    Gas,
    Messages,
    Network,
    Sensor,
    Weather,
    multi_gas_network,
    sensor_gases,
    single_gas_network,
)

# ── columns ────────────────────────────────────────────────────────────────────


def test_enum_members_are_the_column_names():
    df = pd.DataFrame({"CO(GT)": [1.0], "PT08.S4(NO2)": [2.0]})
    assert df[Gas.CO].tolist() == [1.0]
    assert df[Sensor.S4].tolist() == [2.0]
    assert {Gas.CO: 1}[
        "CO(GT)"
    ] == 1  # usable as dict keys interchangeably with strings


def test_latent_names():
    assert Gas.CO.latent == "Gas_CO"
    assert Gas.NOX.latent == "Gas_NOx"
    assert Gas.C6H6.latent == "Gas_C6H6"


# ── CPT ────────────────────────────────────────────────────────────────────────


def test_from_counts_smooths_and_normalizes():
    counts = np.array([[3.0, 1.0], [0.0, 0.0]])  # parent state 0, parent state 1
    cpt = CPT.from_counts("A", ["P"], counts, alpha=1.0)
    np.testing.assert_allclose(cpt.table, [[4 / 6, 2 / 6], [0.5, 0.5]])
    assert cpt.node == "A" and tuple(cpt.parents) == ("P",)


def test_from_counts_unseen_parent_combination_is_uniform_not_nan():
    counts = np.array([[3.0, 1.0], [0.0, 0.0]])
    cpt = CPT.from_counts("A", ["P"], counts, alpha=0.0)
    np.testing.assert_allclose(cpt.table, [[0.75, 0.25], [0.5, 0.5]])


def test_from_counts_root_node():
    cpt = CPT.from_counts("A", [], np.array([2.0, 0.0, 2.0]), alpha=0.0)
    np.testing.assert_allclose(cpt.table, [0.5, 0.0, 0.5])


def test_lookup_reads_one_probability_per_row():
    table = np.array([[0.9, 0.1], [0.3, 0.7]])  # P(A | P)
    cpt = CPT("A", ("P",), table)
    data = pd.DataFrame({"P": [0.0, 1.0, np.nan, 1.0], "A": [1.0, 1.0, 0.0, np.nan]})
    got = cpt.lookup(data)
    np.testing.assert_allclose(got[:2], [0.1, 0.7])
    assert np.isnan(got[2]) and np.isnan(got[3]), "missing node or parent → NaN"


def test_to_frame_has_a_row_per_parent_combination():
    cpt = CPT("A", ("P", "Q"), np.full((2, 3, 4), 0.25))
    frame = cpt.to_frame()
    assert frame.shape == (6, 4)
    np.testing.assert_allclose(frame.sum(axis=1), 1)


# ── Network ────────────────────────────────────────────────────────────────────


def tiny_network(**overrides):
    kwargs = {
        "structure": {
            "B": ("A", "W"),
            "A": ("H",),
            "C": ("A",),
        },  # deliberately not in order
        "cards": {"H": 24, "W": 3, "A": 3, "B": 2, "C": 2},
        "evidence": ("H", "W", "B"),
    }
    return Network(**(kwargs | overrides))


def test_nodes_are_topologically_ordered():
    nodes = tiny_network().nodes
    assert set(nodes) == {"A", "B", "C"}
    assert nodes.index("A") < nodes.index("B") and nodes.index("A") < nodes.index("C")


def test_inputs_observed_children():
    net = tiny_network(latent=("A",), evidence=("H", "W"))
    assert set(net.inputs) == {"H", "W"}
    assert set(net.observed) == {"B", "C"}
    assert set(net.children("A")) == {"B", "C"}
    assert net.children("B") == ()


def test_shape_is_parent_cards_then_node_card():
    assert tiny_network().shape("B") == (3, 3, 2)
    assert tiny_network().shape("A") == (24, 3)


def test_temporal_shape_puts_previous_hour_first():
    net = tiny_network(prev={"A": ("A",)})
    assert net.is_temporal
    assert net.shape("A") == (3, 24, 3)  # A_{t-1}, H, A
    assert not tiny_network().is_temporal


def test_validate_returns_self_when_fine():
    net = tiny_network()
    assert net.validate() is net
    net.validate(columns=["H", "W", "A", "B", "C"])


@pytest.mark.parametrize(
    "overrides, names",
    [
        (
            {"structure": {"A": ("H", "B"), "B": ("A", "W"), "C": ("A",)}},
            r"A|B",
        ),  # A ↔ B cycle
        ({"cards": {"H": 24, "W": 3, "A": 3, "B": 2}}, r"\bC\b"),  # C has no card
        ({"latent": ("A",), "evidence": ("H", "A")}, r"\bA\b"),  # latent in evidence
        ({"prev": {"A": ("ghost",)}}, r"ghost"),  # prev names an unknown node
    ],
)
def test_validate_reports_problems(overrides, names):
    """Each case has exactly one problem; the message should name the culprit."""
    with pytest.raises(ValueError, match=names):
        tiny_network(**overrides).validate()


def test_validate_checks_columns():
    with pytest.raises(ValueError, match=r"\bC\b"):
        tiny_network().validate(columns=["H", "W", "A", "B"])


def test_sensor_gases():
    assert sensor_gases(Sensor.S4, GASES, cross=True) == (Gas.NO2, Gas.CO)
    assert sensor_gases(Sensor.S4, GASES, cross=False) == (Gas.NO2,)
    assert sensor_gases(Sensor.S2, GASES, cross=True) == (Gas.C6H6,)
    assert sensor_gases(Sensor.S2, GASES, cross=False) == ()  # NMHC isn't modeled
    assert sensor_gases(Sensor.S1, [Gas.NO2], cross=True) == ()  # CO not in this model


def test_single_gas_network():
    net = single_gas_network(target=Gas.CO, n_bins=3)
    assert tuple(net.structure[Gas.CO]) == (HOUR,)
    for s in SENSORS:
        assert tuple(net.structure[s]) == (Gas.CO, *WEATHER)
    assert net.cards[HOUR] == 24 and net.cards[Gas.CO] == 3
    assert set(net.evidence) == {HOUR, *WEATHER, *SENSORS}
    assert tuple(net.latent) == ()


def test_single_gas_network_without_weather():
    net = single_gas_network(weather=())
    assert all(tuple(net.structure[s]) == (Gas.CO,) for s in SENSORS)


@pytest.mark.parametrize("cross", [True, False])
def test_multi_gas_network(cross):
    net = multi_gas_network(cross=cross, n_bins=5)
    for g in GASES:
        assert tuple(net.structure[g]) == (HOUR,)
        assert net.cards[g] == 5
    for s in SENSORS:
        assert tuple(net.structure[s]) == (*sensor_gases(s, GASES, cross), *WEATHER)
    assert Gas.NMHC not in net.structure


@pytest.mark.parametrize("build", [single_gas_network, multi_gas_network])
def test_no_reference_is_ever_evidence(build):
    """Evidence is hour, weather and sensors. A reference in evidence leaks the answer."""
    net = build()
    assert not set(net.evidence) & set(Gas)
    assert set(net.evidence) <= {HOUR, *Weather, *Sensor}


# ── FitResult ──────────────────────────────────────────────────────────────────


def test_fit_result_records_iterations():
    r = FitResult()
    assert np.isnan(r.final)
    for v in [-100.0, -50.0, -49.99]:
        r.record(v)
    assert r.n_iter == 3 and r.final == -49.99
    assert list(r.to_frame()["objective"]) == [-100.0, -50.0, -49.99]


def test_should_stop_on_small_relative_gain():
    r = FitResult()
    r.record(-100.0)
    assert not r.should_stop(tol=1e-3), "needs two entries"
    r.record(-50.0)
    assert not r.should_stop(tol=1e-3)
    r.record(-49.99)  # gain 0.01 < 1e-3 × 50
    assert r.should_stop(tol=1e-3)


def test_is_monotone():
    up = FitResult(history=[-10.0, -5.0, -5.0, -4.0])
    down = FitResult(history=[-10.0, -5.0, -6.0])
    assert up.is_monotone() and not down.is_monotone()


# ── Messages ───────────────────────────────────────────────────────────────────
# A 2-state HMM over 3 hours. Every expected value is computed by summing over
# all 2**3 state sequences, not by running forward-backward.

PI = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.2, 0.8]])
E = np.array([[0.9, 0.2], [0.5, 0.5], [0.1, 0.6]])  # E[t, s] = P(evidence_t | s_t)
T, S = 3, 2


def _path_prob(path, upto=T):
    p = PI[path[0]] * E[0, path[0]]
    for t in range(1, upto):
        p *= A[path[t - 1], path[t]] * E[t, path[t]]
    return p


def _brute_messages() -> Messages:
    """alpha, beta, scale straight from their definitions (enumerating paths)."""

    def evidence_prob(upto):
        return sum(
            _path_prob(p, upto) for p in itertools.product(range(S), repeat=upto)
        )

    alpha = np.zeros((T, S))
    for t in range(T):
        for path in itertools.product(range(S), repeat=t + 1):
            alpha[t, path[-1]] += _path_prob(path, t + 1)
        alpha[t] /= alpha[t].sum()
    scale = np.array(
        [evidence_prob(t + 1) / (evidence_prob(t) if t else 1.0) for t in range(T)]
    )
    total = evidence_prob(T)
    gamma = np.zeros((T, S))
    for path in itertools.product(range(S), repeat=T):
        for t in range(T):
            gamma[t, path[t]] += _path_prob(path) / total
    beta = gamma / alpha  # γ = α·β with this scaling
    return Messages(
        alpha=alpha,
        beta=beta,
        scale=scale,
        starts=np.array([True, False, False]),
        loglik=float(np.log(total)),
    )


def test_messages_filtered_and_smoothed():
    m = _brute_messages()
    m.check()
    np.testing.assert_allclose(m.filtered, m.alpha)
    np.testing.assert_allclose(m.smoothed.sum(axis=1), 1)
    np.testing.assert_allclose(
        m.smoothed[-1], m.alpha[-1], err_msg="last hour: smoothed == filtered"
    )


def test_pairwise_matches_enumeration():
    m = _brute_messages()
    total = np.exp(m.loglik)
    for t in (1, 2):
        expected = np.zeros((S, S))
        for path in itertools.product(range(S), repeat=T):
            expected[path[t - 1], path[t]] += _path_prob(path) / total
        np.testing.assert_allclose(m.pairwise(t, A, E[t]), expected, atol=1e-12)


def test_pairwise_undefined_at_segment_start():
    with pytest.raises(ValueError):
        _brute_messages().pairwise(0, A, E[0])


def test_check_catches_bad_shapes():
    m = _brute_messages()
    m.scale = m.scale[:2]
    with pytest.raises(ValueError):
        m.check()
