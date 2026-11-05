"""core/: columns, CPT, FitResult, Messages. Network has its own file (test_network.py)."""

import itertools

import numpy as np
import pandas as pd
import pytest

from cse250a.modules.core import CPT, FitResult, Gas, Messages, Sensor

# ── columns ────────────────────────────────────────────────────────────────────


def test_enum_members_are_the_column_names():
    df = pd.DataFrame({"CO(GT)": [1.0], "PT08.S4(NO2)": [2.0]})
    assert df[Gas.CO].tolist() == [1.0]
    assert df[Sensor.S4].tolist() == [2.0]
    assert {Gas.CO: 1}["CO(GT)"] == 1  # usable as dict keys interchangeably with strings


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
