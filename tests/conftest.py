"""Shared test fixtures: tiny networks, a sampler, and a brute-force reference.

The brute force enumerates every assignment of every unknown node with plain
loops. It is slow (fine for 3–4 nodes) but obviously correct, so when a model
disagrees with it, the model is wrong.

Models are built through `make_bn` / `make_latent`, so if the constructor
signatures change (e.g. to take a Network), only these two helpers change.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from cse250a.modules.core import CPT
from cse250a.modules.models import DiscreteBN, LatentBN

# ── tiny networks ──────────────────────────────────────────────────────────────
# Same shape as the phases: hour → gas → sensors ← weather.
#   H (input) → G → A
#               G → B ← W (input)

STATIC = {"G": ["H"], "A": ["G"], "B": ["G", "W"]}
STATIC_CARDS = {"H": 2, "W": 2, "G": 3, "A": 2, "B": 3}
STATIC_EVIDENCE = ["H", "W", "A", "B"]

# Phase 3 in miniature: Z is a latent true level, R its noisy reference.
#   H → Z → R,  Z → A,  Z → B ← W
LATENT = {"Z": ["H"], "R": ["Z"], "A": ["Z"], "B": ["Z", "W"]}
LATENT_CARDS = {"H": 2, "W": 2, "Z": 2, "R": 2, "A": 2, "B": 3}
LATENT_EVIDENCE = ["H", "W", "A", "B"]


def make_bn(
    structure=STATIC,
    target="G",
    evidence=STATIC_EVIDENCE,
    cards=STATIC_CARDS,
    alpha=1.0,
):
    return DiscreteBN(
        structure, target=target, evidence=evidence, cards=cards, alpha=alpha
    )


def make_latent(
    structure=LATENT,
    target="R",
    evidence=LATENT_EVIDENCE,
    cards=LATENT_CARDS,
    latent=("Z",),
    **kw,
):
    return LatentBN(
        structure,
        target=target,
        evidence=evidence,
        cards=cards,
        latent=list(latent),
        **kw,
    )


def topo(structure):
    order, left = [], dict(structure)
    while left:
        ready = [n for n, ps in left.items() if all(p not in left for p in ps)]
        order += ready
        for n in ready:
            del left[n]
    return order


def shape(structure, cards, node):
    return (*[cards[p] for p in structure[node]], cards[node])


def random_cpts(structure, cards, seed):
    """A CPT per node with random rows (each row drawn from a flat Dirichlet)."""
    rng = np.random.default_rng(seed)
    cpts = {}
    for node, parents in structure.items():
        sh = shape(structure, cards, node)
        table = rng.dirichlet(np.ones(sh[-1]), size=int(np.prod(sh[:-1]))).reshape(sh)
        cpts[node] = CPT(node, tuple(parents), table)
    return cpts


def inputs_of(structure):
    return sorted({p for ps in structure.values() for p in ps} - set(structure))


def sample(structure, cards, cpts, n, seed, missing=0.0, drop=()):
    """Ancestral sample of n rows. Inputs are uniform. Each modeled cell is then
    blanked with probability `missing`; columns in `drop` (latent) are removed."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = {i: int(rng.integers(cards[i])) for i in inputs_of(structure)}
        for node in topo(structure):
            idx = tuple(row[p] for p in structure[node])
            row[node] = int(rng.choice(cards[node], p=cpts[node].table[idx]))
        rows.append(row)
    df = pd.DataFrame(rows).astype(float)
    modeled = [c for c in structure if c not in drop]
    df[modeled] = df[modeled].mask(rng.random((n, len(modeled))) < missing)
    return df.drop(columns=list(drop))


# ── brute force ────────────────────────────────────────────────────────────────


def _assignments(structure, cards, cpts, row):
    """Every completion of one row's unknown nodes, with its probability (unnormalized)."""
    unknown = [n for n in structure if n not in row or pd.isna(row[n])]
    known = {k: int(v) for k, v in row.items() if not pd.isna(v)}
    for states in itertools.product(*[range(cards[n]) for n in unknown]):
        a = {**known, **dict(zip(unknown, states, strict=True))}
        p = math.prod(
            cpts[n].table[tuple(a[q] for q in structure[n]) + (a[n],)]
            for n in structure
        )
        yield a, p


def brute_posterior(structure, cards, cpts, data, node):
    """(n_rows, cards[node]): P(node | observed cells), by enumeration."""
    out = np.zeros((len(data), cards[node]))
    for i, (_, row) in enumerate(data.iterrows()):
        for a, p in _assignments(structure, cards, cpts, row.to_dict()):
            out[i, a[node]] += p
        out[i] /= out[i].sum()
    return out


def brute_loglik(structure, cards, cpts, data):
    return sum(
        math.log(sum(p for _, p in _assignments(structure, cards, cpts, row.to_dict())))
        for _, row in data.iterrows()
    )


def brute_expected_counts(structure, cards, cpts, data):
    """Σ_rows P(node, parents | row) for every node: the lecture's E-step."""
    counts = {n: np.zeros(shape(structure, cards, n)) for n in structure}
    for _, row in data.iterrows():
        comps = list(_assignments(structure, cards, cpts, row.to_dict()))
        z = sum(p for _, p in comps)
        for a, p in comps:
            for n, ps in structure.items():
                counts[n][tuple(a[q] for q in ps) + (a[n],)] += p / z
    return counts


@pytest.fixture
def static_case():
    """(model with random CPTs, data with ~30% of modeled cells missing, true CPTs)."""
    cpts = random_cpts(STATIC, STATIC_CARDS, seed=1)
    data = sample(STATIC, STATIC_CARDS, cpts, 40, seed=2, missing=0.3)
    model = make_bn()
    model.cpts = cpts
    return model, data, cpts
