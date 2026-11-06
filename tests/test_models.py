"""models/: the Model contract, baselines, and DiscreteBN (counting, inference, EM).

A query is (target, evidence), passed at call time, so these tests pass both.
"""

import re

import numpy as np
import pandas as pd
import pytest
from conftest import (
    LATENT,
    LATENT_CARDS,
    LATENT_EVIDENCE,
    STATIC,
    STATIC_CARDS,
    STATIC_EVIDENCE,
    A,
    B,
    G,
    H,
    R,
    W,
    Z,
    brute_expected_counts,
    brute_loglik,
    brute_posterior,
    latent_network,
    make_bn,
    make_latent,
    random_cpts,
    sample,
    shape,
    smoothed,
    static_network,
)

from cse250a.modules.core import TIME_CARDS, Gas, Network, Sensor, Time
from cse250a.modules.models import (
    DiscreteBN,
    HourOnlyBaseline,
    Model,
    PersistenceBaseline,
)

# ── the Model contract ─────────────────────────────────────────────────────────


class Spy(Model):
    """Uniform predictions; remembers what it was handed."""

    def fit(self, train):
        return self

    def _predict_proba(self, data, target):
        self.seen, self.target_seen = data, target
        return pd.DataFrame(1 / 3, index=data.index, columns=range(3))


@pytest.fixture
def spy_data():
    return pd.DataFrame(
        {
            Time.HOUR: [1.0, 2.0],
            Sensor.S1: [0.0, 1.0],
            Gas.CO: [2.0, 0.0],
            Gas.NO2: [1.0, 1.0],
        }
    )


def test_mask_hides_everything_outside_evidence(spy_data):
    spy = Spy()
    spy.predict_proba(spy_data, Gas.CO, (Time.HOUR, Sensor.S1))
    assert spy.seen[[Gas.CO, Gas.NO2]].isna().all().all()
    assert spy.seen[[Time.HOUR, Sensor.S1]].notna().all().all()
    assert spy.target_seen == Gas.CO
    assert spy_data[Gas.CO].notna().all(), "mask must not change the caller's DataFrame"


@pytest.mark.parametrize(
    "evidence",
    [(Time.HOUR, Gas.CO), (Time.HOUR, Gas.NO2)],
    ids=["the target", "another gas's reference"],
)
def test_reference_in_evidence_is_a_leak(spy_data, evidence):
    with pytest.raises(ValueError, match="leak"):
        Spy().predict_proba(spy_data, Gas.CO, evidence)


def test_latent_target_scored_against_its_reference(spy_data):
    """Phase 3: query the latent node, score against the reference column."""
    result = Spy().evaluate(spy_data, Gas.CO.latent, (Time.HOUR,), against=Gas.CO)
    assert result["n"] == 2


def test_requires_is_checked(spy_data):
    class NeedsHour(Spy):
        requires = (Time.HOUR,)

    with pytest.raises(ValueError, match="hour"):
        NeedsHour().predict_proba(spy_data, Gas.CO, (Sensor.S1,))


def test_privileged_skips_masking(spy_data):
    class Peeks(Spy):
        privileged = True

    spy = Peeks()
    spy.predict_proba(spy_data, Gas.CO, (Time.HOUR,))
    assert spy.seen[Gas.CO].notna().all(), "privileged models see the target"


def test_evaluate_predicts_every_row_then_scores_observed_ones():
    data = pd.DataFrame(
        {Time.HOUR: [0.0, 1.0, 2.0, 3.0], Gas.CO: [0.0, np.nan, 2.0, 1.0]}
    )
    spy = Spy()
    result = spy.evaluate(data, Gas.CO, (Time.HOUR,))
    assert len(spy.seen) == 4, "the HMM needs the whole sequence"
    assert result["n"] == 3


# ── baselines ──────────────────────────────────────────────────────────────────

BASE_CARDS = {Gas.CO: 2, Gas.NO2: 2, **TIME_CARDS}


def test_hour_only_baseline_is_smoothed_counts_per_hour():
    train = pd.DataFrame(
        {Time.HOUR: [0, 0, 0, 1, 1], Gas.CO: [0, 0, 1, 1, np.nan]}, dtype=float
    )
    model = HourOnlyBaseline(cards={Gas.CO: 2, **TIME_CARDS}, alpha=1.0).fit(train)
    test = pd.DataFrame({Time.HOUR: [0.0, 1.0], Gas.CO: [1.0, 0.0]})
    proba = model.predict_proba(test, Gas.CO, (Time.HOUR,))
    np.testing.assert_allclose(proba.to_numpy(), [[3 / 5, 2 / 5], [1 / 3, 2 / 3]])


def test_hour_only_baseline_one_fit_answers_every_column():
    train = pd.DataFrame(
        {Time.HOUR: [0, 1, 0, 1], Gas.CO: [0, 1, 0, 1], Gas.NO2: [1, 1, 0, 0]},
        dtype=float,
    )
    model = HourOnlyBaseline(cards=BASE_CARDS).fit(train)
    for target in (Gas.CO, Gas.NO2):
        assert model.evaluate(train, target, (Time.HOUR,))["n"] == 4


def test_hour_only_baseline_ignores_extra_evidence():
    train = pd.DataFrame(
        {Time.HOUR: [0, 1, 0, 1], Gas.CO: [0, 1, 0, 1], Sensor.S1: [0, 1, 1, 0]},
        dtype=float,
    )
    model = HourOnlyBaseline(cards={Gas.CO: 2, **TIME_CARDS}).fit(train)
    narrow = model.predict_proba(train, Gas.CO, (Time.HOUR,))
    wide = model.predict_proba(train, Gas.CO, (Time.HOUR, Sensor.S1))
    pd.testing.assert_frame_equal(narrow, wide)


def test_hour_only_baseline_needs_hour_in_evidence():
    train = pd.DataFrame({Time.HOUR: [0.0, 1.0], Gas.CO: [0.0, 1.0]})
    model = HourOnlyBaseline(cards={Gas.CO: 2, **TIME_CARDS}).fit(train)
    with pytest.raises(ValueError, match="hour"):
        model.predict_proba(train, Gas.CO, (Sensor.S1,))


def test_hour_only_baseline_unfitted_target():
    train = pd.DataFrame({Time.HOUR: [0.0, 1.0], Gas.CO: [0.0, 1.0]})
    model = HourOnlyBaseline(cards={Gas.CO: 2, **TIME_CARDS}).fit(train)
    with pytest.raises(KeyError, match=re.escape("NO2(GT)")):
        model.predict_proba(train.assign(**{Gas.NO2: 0.0}), Gas.NO2, (Time.HOUR,))


def test_hour_only_baseline_rejects_raw_values():
    train = pd.DataFrame({Time.HOUR: [0.0, 1.0], Gas.CO: [0.4, 2.7]})
    with pytest.raises(ValueError, match="Discretizer"):
        HourOnlyBaseline(cards={Gas.CO: 3, **TIME_CARDS}).fit(train)


def test_persistence_previous_respects_gaps():
    index = pd.to_datetime(
        ["2004-03-10 18:00", "2004-03-10 19:00", "2004-03-10 21:00", "2004-03-10 22:00"]
    )
    data = pd.DataFrame(
        {Time.HOUR: index.hour.astype(float), Gas.CO: [0.0, 1.0, 2.0, np.nan]},
        index=index,
    )
    prev = PersistenceBaseline(cards=BASE_CARDS)._previous(data, Gas.CO)
    assert np.isnan(prev.iloc[0]), "nothing before the first hour"
    assert prev.iloc[1] == 0.0
    assert np.isnan(prev.iloc[2]), "20:00 is missing, so 21:00 has no previous hour"
    assert prev.iloc[3] == 2.0


def test_persistence_is_privileged():
    assert PersistenceBaseline.privileged
    assert Time.HOUR in PersistenceBaseline.requires, "the fallback needs the hour"


# ── DiscreteBN: construction ───────────────────────────────────────────────────


def test_bn_takes_a_network_and_no_query():
    bn = make_bn()
    assert bn.network == static_network()
    assert not hasattr(bn, "target")


def test_bn_requires_every_input():
    """Inputs have no CPT, so a query can't marginalize them out."""
    assert set(make_bn().requires) == {H, W}


def test_bn_refuses_temporal_networks():
    with pytest.raises(NotImplementedError):
        make_bn(static_network(prev={G: (G,)}))


def test_bn_validates_on_construction():
    with pytest.raises(ValueError, match="hr"):
        DiscreteBN(Network({G: ("hr",)}, {G: 3, "hr": 24}))


# ── DiscreteBN: fit checks ─────────────────────────────────────────────────────


def test_fit_rejects_a_node_that_is_not_a_column_and_not_latent():
    """A misspelled or missing column must raise, not silently become hidden."""
    data = sample(STATIC, STATIC_CARDS, random_cpts(STATIC, STATIC_CARDS, 0), 20, 1)
    with pytest.raises(ValueError, match=re.escape(A)):
        make_bn().fit(data.drop(columns=[A]))


def test_fit_rejects_missing_inputs():
    data = sample(STATIC, STATIC_CARDS, random_cpts(STATIC, STATIC_CARDS, 0), 20, 1)
    data.loc[:2, W] = np.nan
    with pytest.raises(ValueError, match="3 rows"):
        make_bn().fit(data)


def test_init_with_the_wrong_shape_is_caught():
    data = sample(STATIC, STATIC_CARDS, random_cpts(STATIC, STATIC_CARDS, 0), 20, 1)
    with pytest.raises(ValueError, match=re.escape(A)):
        make_bn(init={A: np.full((3, 3), 1 / 3)}).fit(data)


# ── DiscreteBN: counting and inference ─────────────────────────────────────────


def test_complete_counts_are_complete_case():
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=3)
    data = sample(STATIC, STATIC_CARDS, cpts, 300, seed=4, missing=0.2)
    counts = make_bn()._complete_counts(data)
    assert set(counts) == set(STATIC)
    expected = np.zeros(shape(STATIC, STATIC_CARDS, B))  # B | G, W
    for g, w, b in data[[G, W, B]].dropna().astype(int).itertuples(index=False):
        expected[g, w, b] += 1
    np.testing.assert_array_equal(counts[B], expected)


def test_fit_on_complete_data_is_smoothed_counts_in_one_step():
    """Nothing missing → EM's first step is plain counting, and it stops there."""
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=5)
    data = sample(STATIC, STATIC_CARDS, cpts, 300, seed=6)
    model = make_bn(alpha=1.0).fit(data)
    counts = model._complete_counts(data)
    for node in STATIC:
        np.testing.assert_allclose(
            model.cpts[node].table, smoothed(counts[node], 1.0), err_msg=node
        )
    assert model.result.converged
    assert model.result.n_iter <= 2


def test_fit_recovers_the_true_cpts_with_lots_of_data():
    truth = random_cpts(STATIC, STATIC_CARDS, seed=7)
    data = sample(STATIC, STATIC_CARDS, truth, 20_000, seed=8)
    model = make_bn().fit(data)
    for node in STATIC:
        np.testing.assert_allclose(
            model.cpts[node].table, truth[node].table, atol=0.05, err_msg=node
        )


def test_joint_is_a_distribution_over_the_unknowns(static_case):
    model, data, _ = static_case
    row = data.iloc[0].copy()
    row[[G, A]] = np.nan
    hidden, joint = model._joint(row, keep_all=True)
    assert {G, A} <= set(hidden)
    assert joint.shape == tuple(STATIC_CARDS[h] for h in hidden)
    assert joint.sum() == pytest.approx(1.0)


def test_posterior_matches_brute_force(static_case):
    """Rows have random gaps: the target, sensors, or both can be missing."""
    model, data, cpts = static_case
    for node in [G, A]:
        np.testing.assert_allclose(
            model.posterior(data, node),
            brute_posterior(STATIC, STATIC_CARDS, cpts, data, node),
            atol=1e-10,
        )


def test_posterior_of_an_observed_node_is_one_hot(static_case):
    model, data, _ = static_case
    seen = data[G].notna().to_numpy()
    post = model.posterior(data, G)[seen]
    np.testing.assert_array_equal(post.argmax(axis=1), data.loc[seen, G].astype(int))
    np.testing.assert_allclose(post.max(axis=1), 1)


def test_predict_proba_shape_and_masking(static_case):
    model, data, _ = static_case
    proba = model.predict_proba(data, G, STATIC_EVIDENCE)
    assert proba.index.equals(data.index)
    assert list(proba.columns) == list(range(STATIC_CARDS[G]))
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    scrambled = data.assign(
        **{G: np.random.default_rng(0).integers(0, 3, len(data)).astype(float)}
    )
    pd.testing.assert_frame_equal(
        proba, model.predict_proba(scrambled, G, STATIC_EVIDENCE), obj="target hidden"
    )


def test_one_fit_answers_different_queries(static_case):
    """Target and evidence are per query: predict a sensor, or drop a sensor."""
    model, data, cpts = static_case
    hidden_a = data.assign(**{A: np.nan, G: np.nan})
    np.testing.assert_allclose(
        model.predict_proba(data, A, (H, W, B)).to_numpy(),
        brute_posterior(STATIC, STATIC_CARDS, cpts, hidden_a, A),
        atol=1e-10,
    )
    without_b = model.predict_proba(data, G, (H, W, A))
    np.testing.assert_allclose(without_b.sum(axis=1), 1)
    assert not np.allclose(without_b, model.predict_proba(data, G, STATIC_EVIDENCE))


def test_query_without_an_input_is_refused(static_case):
    model, data, _ = static_case
    with pytest.raises(ValueError, match=re.escape(W)):
        model.predict_proba(data, G, (H, A, B))


def test_log_likelihood_matches_brute_force(static_case):
    model, data, cpts = static_case
    assert model.log_likelihood(data) == pytest.approx(
        brute_loglik(STATIC, STATIC_CARDS, cpts, data)
    )


def test_sparsity_reports_every_cpt():
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=9)
    data = sample(STATIC, STATIC_CARDS, cpts, 50, seed=10)
    report = make_bn().fit(data).sparsity(data)
    assert len(report) == len(STATIC)
    assert {"empty", "thin"} <= set(report.columns)


# ── DiscreteBN: EM (missing values and latent nodes) ───────────────────────────


def test_expected_counts_on_complete_data_are_complete_counts():
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=11)
    data = sample(STATIC, STATIC_CARDS, cpts, 60, seed=12)
    model = make_bn()
    model.cpts = cpts
    soft, _ = model._expected_counts(data)
    hard = model._complete_counts(data)
    for node in STATIC:
        np.testing.assert_allclose(soft[node], hard[node], err_msg=node)


def test_expected_counts_and_loglik_match_brute_force():
    """Latent Z has no column at all: the BN must treat it as hidden on every row."""
    cpts = random_cpts(LATENT, LATENT_CARDS, seed=13)
    data = sample(LATENT, LATENT_CARDS, cpts, 30, seed=14, missing=0.3, drop=[Z])
    assert Z not in data.columns
    model = make_latent()
    model.cpts = cpts
    got, loglik = model._expected_counts(data)
    expected = brute_expected_counts(LATENT, LATENT_CARDS, cpts, data)
    for node in LATENT:
        np.testing.assert_allclose(got[node], expected[node], atol=1e-10, err_msg=node)
    assert loglik == pytest.approx(brute_loglik(LATENT, LATENT_CARDS, cpts, data))


def test_em_uses_rows_with_missing_values():
    """A missing G with A and B observed still carries information, so EM's CPTs
    differ from counting complete rows only."""
    truth = random_cpts(STATIC, STATIC_CARDS, seed=21)
    data = sample(STATIC, STATIC_CARDS, truth, 400, seed=22, missing=0.3)
    model = make_bn(alpha=1.0).fit(data)
    complete = model._complete_counts(data)
    assert not np.allclose(model.cpts[G].table, smoothed(complete[G], 1.0))
    assert model.result.n_iter > 2, "missing data should take more than one step"


@pytest.mark.parametrize("which", ["missing", "latent"])
def test_em_objective_never_decreases(which):
    if which == "missing":
        truth = random_cpts(STATIC, STATIC_CARDS, seed=17)
        data = sample(STATIC, STATIC_CARDS, truth, 400, seed=18, missing=0.3)
        model = make_bn(max_iter=30, tol=0.0).fit(data)
    else:
        truth = random_cpts(LATENT, LATENT_CARDS, seed=17)
        data = sample(LATENT, LATENT_CARDS, truth, 400, seed=18, missing=0.3, drop=[Z])
        model = make_latent(max_iter=30, tol=0.0).fit(data)
    h = np.asarray(model.result.history)
    assert len(h) >= 2
    assert model.result.is_monotone(), "a drop means a bug in the E- or M-step"


def test_init_cpts_are_random_seeded_and_respect_overrides():
    """Every LATENT family touches Z, so every start is random (seeded); `init`
    overrides a node's start."""
    data = sample(
        LATENT, LATENT_CARDS, random_cpts(LATENT, LATENT_CARDS, 0), 50, 1, drop=[Z]
    )
    anchor = np.array([[0.8, 0.2], [0.2, 0.8]])
    a = make_latent(init={R: anchor}, seed=0)._init_cpts(data)
    b = make_latent(init={R: anchor}, seed=0)._init_cpts(data)
    c = make_latent(init={R: anchor}, seed=1)._init_cpts(data)
    assert set(a) == set(LATENT)
    np.testing.assert_allclose(a[R].table, anchor)
    for node in LATENT:
        np.testing.assert_allclose(a[node].table.sum(axis=-1), 1, err_msg=node)
        np.testing.assert_allclose(
            a[node].table, b[node].table, err_msg="same seed, same start"
        )
    assert not np.allclose(a[Z].table, c[Z].table), "different seed, different start"


def test_init_cpts_use_complete_counts_where_nothing_is_latent():
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=23)
    data = sample(STATIC, STATIC_CARDS, cpts, 200, seed=24, missing=0.2)
    model = make_bn(alpha=1.0)
    start = model._init_cpts(data)
    complete = model._complete_counts(data)
    for node in STATIC:
        np.testing.assert_allclose(
            start[node].table, smoothed(complete[node], 1.0), err_msg=node
        )


def test_latent_posterior_matches_brute_force():
    cpts = random_cpts(LATENT, LATENT_CARDS, seed=19)
    data = sample(LATENT, LATENT_CARDS, cpts, 30, seed=20, missing=0.3, drop=[Z])
    model = make_latent()
    model.cpts = cpts
    np.testing.assert_allclose(
        model.posterior(data, Z),
        brute_posterior(LATENT, LATENT_CARDS, cpts, data, Z),
        atol=1e-10,
    )


def test_latent_query_scored_against_the_reference():
    """Phase 3: P(Z | evidence) scored against R, the reference reading."""
    truth = random_cpts(LATENT, LATENT_CARDS, seed=25)
    data = sample(LATENT, LATENT_CARDS, truth, 200, seed=26, missing=0.3, drop=[Z])
    model = make_latent(latent_network(), max_iter=10).fit(data)
    result = model.evaluate(data, Z, LATENT_EVIDENCE, against=R)
    assert result["n"] == data[R].notna().sum()


def test_latent_parent_of_several_gases():
    """Phase 3b: a hidden Event drives two gases. EM must stay monotone and give
    a posterior over Event from sensors alone."""
    no2 = Gas.NO2
    structure = {"Event": (H,), G: ("Event",), no2: ("Event",), A: (G,), B: (no2, G)}
    cards = {H: 2, "Event": 2, G: 3, no2: 3, A: 3, B: 3}
    truth = random_cpts(structure, cards, seed=27)
    data = sample(structure, cards, truth, 300, seed=28, missing=0.3, drop=["Event"])
    net = Network(structure, cards, latent=("Event",)).validate(data.columns)
    model = DiscreteBN(net, max_iter=20).fit(data)
    assert model.result.is_monotone()
    post = model.predict_proba(data, "Event", (H, A, B))
    np.testing.assert_allclose(post.sum(axis=1), 1)
