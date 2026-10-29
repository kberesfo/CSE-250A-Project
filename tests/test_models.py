"""models/: the Model contract, baselines, DiscreteBN (Phases 1–2), LatentBN (Phase 3)."""

import numpy as np
import pandas as pd
import pytest
from conftest import (
    LATENT,
    LATENT_CARDS,
    STATIC,
    STATIC_CARDS,
    brute_expected_counts,
    brute_loglik,
    brute_posterior,
    make_bn,
    make_latent,
    random_cpts,
    sample,
    shape,
)

from cse250a.modules.models import HourOnlyBaseline, Model, PersistenceBaseline

# ── the Model contract (implemented: these pass now) ───────────────────────────


class Spy(Model):
    """Uniform predictions; remembers what it was handed."""

    def fit(self, train):
        return self

    def _predict_proba(self, data):
        self.seen = data
        return pd.DataFrame(1 / 3, index=data.index, columns=range(3))


def test_mask_hides_everything_outside_evidence():
    data = pd.DataFrame(
        {
            "hour": [1.0, 2.0],
            "S": [0.0, 1.0],
            "target": [2.0, 0.0],
            "other_ref": [1.0, 1.0],
        }
    )
    spy = Spy("target", evidence=["hour", "S"])
    spy.predict_proba(data)
    assert spy.seen[["target", "other_ref"]].isna().all().all()
    assert spy.seen[["hour", "S"]].notna().all().all()
    assert data["target"].notna().all(), "mask must not change the caller's DataFrame"


def test_evaluate_predicts_every_row_then_scores_observed_ones():
    data = pd.DataFrame(
        {"hour": [0.0, 1.0, 2.0, 3.0], "target": [0.0, np.nan, 2.0, 1.0]}
    )
    spy = Spy("target", evidence=["hour"])
    result = spy.evaluate(data)
    assert len(spy.seen) == 4, "the HMM needs the whole sequence"
    assert result["n"] == 3


# ── baselines ──────────────────────────────────────────────────────────────────


def test_hour_only_baseline_is_smoothed_counts_per_hour():
    train = pd.DataFrame(
        {"hour": [0, 0, 0, 1, 1], "y": [0, 0, 1, 1, np.nan]}, dtype=float
    )
    model = HourOnlyBaseline("y", alpha=1.0).fit(train)
    proba = model.predict_proba(pd.DataFrame({"hour": [0.0, 1.0], "y": [1.0, 0.0]}))
    np.testing.assert_allclose(proba.to_numpy(), [[3 / 5, 2 / 5], [1 / 3, 2 / 3]])


def test_persistence_previous_respects_gaps():
    index = pd.to_datetime(
        ["2004-03-10 18:00", "2004-03-10 19:00", "2004-03-10 21:00", "2004-03-10 22:00"]
    )
    data = pd.DataFrame(
        {"hour": index.hour.astype(float), "y": [0.0, 1.0, 2.0, np.nan]}, index=index
    )
    prev = PersistenceBaseline("y")._previous(data)
    assert np.isnan(prev.iloc[0]), "nothing before the first hour"
    assert prev.iloc[1] == 0.0
    assert np.isnan(prev.iloc[2]), "20:00 is missing, so 21:00 has no previous hour"
    assert prev.iloc[3] == 2.0


# ── DiscreteBN (Phases 1–2) ────────────────────────────────────────────────────


def test_counts_are_complete_case():
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=3)
    data = sample(STATIC, STATIC_CARDS, cpts, 300, seed=4, missing=0.2)
    counts = make_bn()._counts(data)
    assert set(counts) == set(STATIC)
    expected = np.zeros(shape(STATIC, STATIC_CARDS, "B"))  # B | G, W
    for g, w, b in data[["G", "W", "B"]].dropna().astype(int).itertuples(index=False):
        expected[g, w, b] += 1
    np.testing.assert_array_equal(counts["B"], expected)


def test_fit_is_smoothed_normalized_counts():
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=5)
    data = sample(STATIC, STATIC_CARDS, cpts, 300, seed=6, missing=0.2)
    model = make_bn(alpha=1.0).fit(data)
    counts = model._counts(data)
    for node in STATIC:
        expected = (counts[node] + 1.0) / (counts[node] + 1.0).sum(
            axis=-1, keepdims=True
        )
        np.testing.assert_allclose(model.cpts[node].table, expected, err_msg=node)


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
    row[["G", "A"]] = np.nan
    hidden, joint = model._joint(row, keep_all=True)
    assert {"G", "A"} <= set(hidden)
    assert joint.shape == tuple(STATIC_CARDS[h] for h in hidden)
    assert joint.sum() == pytest.approx(1.0)


def test_posterior_matches_brute_force(static_case):
    """Rows have random gaps: the target, sensors, or both can be missing."""
    model, data, cpts = static_case
    for node in ["G", "A"]:
        np.testing.assert_allclose(
            model.posterior(data, node),
            brute_posterior(STATIC, STATIC_CARDS, cpts, data, node),
            atol=1e-10,
        )


def test_posterior_of_an_observed_node_is_one_hot(static_case):
    model, data, _ = static_case
    seen = data["G"].notna().to_numpy()
    post = model.posterior(data, "G")[seen]
    np.testing.assert_array_equal(post.argmax(axis=1), data.loc[seen, "G"].astype(int))
    np.testing.assert_allclose(post.max(axis=1), 1)


def test_predict_proba_shape_and_masking(static_case):
    model, data, _ = static_case
    proba = model.predict_proba(data)
    assert proba.index.equals(data.index)
    assert list(proba.columns) == list(range(STATIC_CARDS["G"]))
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    scrambled = data.assign(
        G=np.random.default_rng(0).integers(0, 3, len(data)).astype(float)
    )
    pd.testing.assert_frame_equal(
        proba, model.predict_proba(scrambled), obj="target must be hidden"
    )


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


# ── LatentBN (Phase 3) ─────────────────────────────────────────────────────────


def history(model):
    """EM objective per iteration, whether stored as a list or a FitResult."""
    result = getattr(model, "result", None)
    return np.asarray(result.history if result is not None else model.history)


def test_expected_counts_on_complete_data_are_hard_counts():
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=11)
    data = sample(STATIC, STATIC_CARDS, cpts, 60, seed=12)
    em = make_latent(
        structure=STATIC,
        target="G",
        evidence=["H", "W", "A", "B"],
        cards=STATIC_CARDS,
        latent=(),
    )
    em.cpts = cpts
    soft, hard = em._counts(data), make_bn()._counts(data)
    for node in STATIC:
        np.testing.assert_allclose(soft[node], hard[node], err_msg=node)


def test_expected_counts_match_brute_force():
    cpts = random_cpts(LATENT, LATENT_CARDS, seed=13)
    data = sample(LATENT, LATENT_CARDS, cpts, 30, seed=14, missing=0.3, drop=["Z"])
    em = make_latent()
    em.cpts = cpts
    got = em._counts(data.assign(Z=np.nan))
    expected = brute_expected_counts(LATENT, LATENT_CARDS, cpts, data)
    for node in LATENT:
        np.testing.assert_allclose(got[node], expected[node], atol=1e-10, err_msg=node)


def test_em_on_complete_data_equals_mle():
    truth = random_cpts(STATIC, STATIC_CARDS, seed=15)
    data = sample(STATIC, STATIC_CARDS, truth, 500, seed=16)
    mle = make_bn().fit(data)
    em = make_latent(
        structure=STATIC,
        target="G",
        evidence=["H", "W", "A", "B"],
        cards=STATIC_CARDS,
        latent=(),
        max_iter=5,
    )
    em.fit(data)
    for node in STATIC:
        np.testing.assert_allclose(
            em.cpts[node].table, mle.cpts[node].table, err_msg=node
        )


def test_em_objective_never_decreases():
    truth = random_cpts(LATENT, LATENT_CARDS, seed=17)
    data = sample(LATENT, LATENT_CARDS, truth, 400, seed=18, missing=0.3, drop=["Z"])
    em = make_latent(max_iter=30, tol=0.0).fit(data)
    h = history(em)
    assert len(h) >= 2
    assert np.all(np.diff(h) >= -1e-8 * np.abs(h[:-1])), (
        "a drop means a bug in the E- or M-step"
    )


def test_init_cpts_are_random_seeded_and_respect_overrides():
    anchor = np.array([[0.8, 0.2], [0.2, 0.8]])
    a = make_latent(init={"R": anchor}, seed=0)._init_cpts()
    b = make_latent(init={"R": anchor}, seed=0)._init_cpts()
    c = make_latent(init={"R": anchor}, seed=1)._init_cpts()
    assert set(a) == set(LATENT)
    np.testing.assert_allclose(a["R"].table, anchor)
    for node in LATENT:
        np.testing.assert_allclose(a[node].table.sum(axis=-1), 1, err_msg=node)
        np.testing.assert_allclose(
            a[node].table, b[node].table, err_msg="same seed, same start"
        )
    assert not np.allclose(a["Z"].table, c["Z"].table), (
        "different seed, different start"
    )


def test_latent_posterior_matches_brute_force():
    cpts = random_cpts(LATENT, LATENT_CARDS, seed=19)
    data = sample(LATENT, LATENT_CARDS, cpts, 30, seed=20, missing=0.3, drop=["Z"])
    em = make_latent()
    em.cpts = cpts
    np.testing.assert_allclose(
        em.posterior(data, "Z"),
        brute_posterior(LATENT, LATENT_CARDS, cpts, data, "Z"),
        atol=1e-10,
    )
