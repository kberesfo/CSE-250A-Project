# modules/models/layered.py
"""A vectorized DiscreteBN for the layered networks of Phases 2 and 3b.

DiscreteBN's E-step loops over rows in Python and enumerates each row's hidden
states one at a time. That is exact and general, but a Phase 3b fit (Event + 4
gases + 5 sensors, ~4,500 rows) takes ~3 s per EM iteration, and the #19 grid
needs 180 fits. LayeredBN computes the same quantities with numpy arrays over all
rows at once.

It is a drop-in subclass: `fit`, the M-step, initialization, the MAP objective
and `predict_proba` are DiscreteBN's. Only three methods are replaced:

    _expected_counts   (the E-step)
    log_likelihood
    posterior

Supported structure (raises otherwise). Every modeled node is in one of three
layers, and inputs (hour, weekday, weather) may be parents anywhere:

    top     at most one latent node, parents = inputs only        (Event)
    middle  parents ⊆ {top} ∪ inputs                              (gases)
    leaf    parents ⊆ middle ∪ inputs, no children                (sensors)

Phase 2 (hour → gas → sensor) is the same thing with no top node.

How it works: for every row, enumerate all K × C joint states of (top, middle),
where C = Π cards of the middle nodes (3^4 = 81 for four gases at 3 bins). The
joint is the product of every CPT entry, observed middle nodes are enforced by a
0/1 mask, and an observed leaf contributes its CPT entry; a missing leaf sums out
to 1. That gives a (rows, K, C) table whose row sums are P(observed cells), the
same normalizer DiscreteBN uses. Expected counts are marginals of that table.
tests/test_layered.py checks counts, log-likelihood, posteriors and EM histories
against DiscreteBN on real data.
"""

import itertools

import numpy as np
import pandas as pd

from cse250a.modules.core import Network

from .bn import DiscreteBN


class LayeredBN(DiscreteBN):
    """DiscreteBN with a vectorized E-step for top → middle → leaf networks."""

    name = "layered_bn"

    def __init__(self, network: Network, **kwargs):
        super().__init__(network, **kwargs)
        self._layers()

    # ── structure ───────────────────────────────────────────────────────────

    def _layers(self) -> None:
        net = self.network
        modeled = set(net.structure)
        if len(net.latent) > 1:
            raise ValueError(f"{self.name}: at most one latent node, got {net.latent}")
        self.top = net.latent[0] if net.latent else None
        if self.top is not None and any(p in modeled for p in net.structure[self.top]):
            raise ValueError(
                f"{self.name}: latent {self.top} must have only inputs as parents"
            )

        allowed_mid = {self.top} if self.top else set()
        self.middle = tuple(
            n
            for n in net.structure
            if n != self.top
            and all(p not in modeled or p in allowed_mid for p in net.structure[n])
        )
        mid = set(self.middle)
        self.leaves = tuple(n for n in net.structure if n != self.top and n not in mid)
        for leaf in self.leaves:
            bad = [p for p in net.structure[leaf] if p in modeled and p not in mid]
            if bad or net.children(leaf):
                raise ValueError(
                    f"{self.name}: {leaf} must have only middle nodes and inputs as "
                    f"parents and no children (bad parents: {bad})"
                )
        for m in self.middle:
            if any(p in self.leaves for p in net.structure[m]):
                raise ValueError(f"{self.name}: {m} has a leaf parent")

        self.K = net.cards[self.top] if self.top else 1
        self.combos = np.array(
            list(itertools.product(*[range(net.cards[m]) for m in self.middle])),
            dtype=int,
        ).reshape(-1, len(self.middle))  # (C, n_middle)

    # ── the (rows, K, C) joint ──────────────────────────────────────────────

    def _cpt_rows(
        self, node: str, data: dict[str, np.ndarray], free: dict[str, np.ndarray]
    ):
        """Gather CPT entries for every row.

        `free` maps some parents/the node itself to index arrays that broadcast
        against the row axis; every other parent is read from `data` (one value
        per row, shape (n, 1, ...)). Returns table[index] with broadcasting.
        """
        table = self.cpts[node].table
        idx = []
        for col in self.network.family(node):
            idx.append(free[col] if col in free else data[col])
        return table[tuple(idx)]

    def _joint_table(self, df: pd.DataFrame):
        """Unnormalized joint over (top, middle) for every row: (n, K, C), plus
        the pieces the E-step reuses."""
        net = self.network
        n = len(df)
        C = len(self.combos)
        cols = {}
        for c in (*net.inputs, *self.middle, *self.leaves):
            if c in df.columns:
                v = df[c].to_numpy(dtype=float)
            else:
                v = np.full(n, np.nan)
            cols[c] = v
        inputs = {c: cols[c].astype(int)[:, None, None] for c in net.inputs}

        k_axis = np.arange(self.K)[None, :, None]  # (1, K, 1)
        joint = np.ones((n, self.K, C))

        # top: P(top | inputs) → (n, K, 1)
        if self.top:
            joint *= self._cpt_rows(self.top, inputs, {self.top: k_axis})

        # middle: P(m | top, inputs) for the combo's value of m → (n, K, C)
        mid_obs = {}
        for j, m in enumerate(self.middle):
            val = self.combos[:, j][None, None, :]  # (1, 1, C)
            free = {m: val}
            if self.top:
                free[self.top] = k_axis
            joint *= self._cpt_rows(m, inputs, free)
            obs = cols[m]
            seen = ~np.isnan(obs)
            mid_obs[m] = (obs, seen)
            mask = np.ones((n, C))
            mask[seen] = self.combos[:, j][None, :] == obs[seen][:, None]
            joint *= mask[:, None, :]

        # leaves: observed → P(leaf = obs | middle combo, inputs); missing → 1
        leaf_tabs = {}
        for leaf in self.leaves:
            free = {m: self.combos[:, j][None, :] for j, m in enumerate(self.middle)}
            inp2 = {c: v[:, :, 0] for c, v in inputs.items()}  # (n, 1)
            full = self._cpt_rows(
                leaf, inp2, {**free, leaf: slice(None)}
            )  # (n, C, card)
            full = np.broadcast_to(full, (n, C, net.cards[leaf]))
            obs = cols[leaf]
            seen = ~np.isnan(obs)
            factor = np.ones((n, C))
            factor[seen] = np.take_along_axis(
                full[seen],
                obs[seen].astype(int)[:, None, None].repeat(C, axis=1),
                axis=2,
            )[:, :, 0]
            joint *= factor[:, None, :]
            leaf_tabs[leaf] = (full, obs, seen)

        return joint, inputs, mid_obs, leaf_tabs

    # ── DiscreteBN overrides ────────────────────────────────────────────────

    def _expected_counts(
        self, data: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], float]:
        net = self.network
        joint, inputs, _, leaf_tabs = self._joint_table(data)
        z = joint.sum(axis=(1, 2))
        loglik = float(np.log(z).sum())
        post = joint / z[:, None, None]  # (n, K, C)
        counts = {node: np.zeros(net.shape(node)) for node in net.structure}
        rows = {c: v[:, 0, 0] for c, v in inputs.items()}

        def add(node, fixed: dict[str, int | np.ndarray], weights: np.ndarray):
            """counts[node][family index] += weights, family values from `fixed` or rows."""
            family = net.family(node)
            if all(c in fixed for c in family):  # no per-row index: one cell
                counts[node][tuple(fixed[c] for c in family)] += weights.sum()
                return
            idx = tuple(fixed[c] if c in fixed else rows[c] for c in family)
            np.add.at(counts[node], idx, weights)

        p_top = post.sum(axis=2)  # (n, K)
        if self.top:
            for k in range(self.K):
                add(self.top, {self.top: k}, p_top[:, k])

        for j, m in enumerate(self.middle):
            for v in range(net.cards[m]):
                sel = self.combos[:, j] == v
                w = post[:, :, sel].sum(axis=2)  # (n, K)
                for k in range(self.K):
                    fixed = {m: v}
                    if self.top:
                        fixed[self.top] = k
                    add(m, fixed, w[:, k])

        q = post.sum(axis=1)  # (n, C): posterior over middle combos
        for leaf in self.leaves:
            full, obs, seen = leaf_tabs[leaf]
            card = net.cards[leaf]
            # P(combo, leaf value | row): one-hot if observed, else q · CPT
            pl = np.empty((len(q), q.shape[1], card))
            pl[~seen] = q[~seen][:, :, None] * full[~seen]
            oh = np.zeros((int(seen.sum()), card))
            oh[np.arange(len(oh)), obs[seen].astype(int)] = 1.0
            pl[seen] = q[seen][:, :, None] * oh[:, None, :]
            mids = [p for p in net.structure[leaf] if p in self.middle]
            jidx = [self.middle.index(p) for p in mids]
            configs = sorted({tuple(c) for c in self.combos[:, jidx]}) if mids else [()]
            for cfg in configs:
                sel = (
                    np.all(self.combos[:, jidx] == np.array(cfg), axis=1)
                    if mids
                    else np.ones(len(self.combos), bool)
                )
                w = pl[:, sel, :].sum(axis=1)  # (n, card)
                for v in range(card):
                    add(leaf, {**dict(zip(mids, cfg)), leaf: v}, w[:, v])

        return counts, loglik

    def log_likelihood(self, data: pd.DataFrame) -> float:
        if not self.cpts:
            raise RuntimeError(f"{self.name}: call fit() before scoring")
        joint, *_ = self._joint_table(data)
        return float(np.log(joint.sum(axis=(1, 2))).sum())

    def row_log_likelihood(self, data: pd.DataFrame) -> pd.Series:
        """log P(observed cells) per row (same normalizer as log_likelihood)."""
        joint, *_ = self._joint_table(data)
        return pd.Series(np.log(joint.sum(axis=(1, 2))), index=data.index)

    def posterior(self, data: pd.DataFrame, node: str) -> np.ndarray:
        if not self.cpts:
            raise RuntimeError(f"{self.name}: call fit() before predicting")
        net = self.network
        joint, _, _, leaf_tabs = self._joint_table(data)
        post = joint / joint.sum(axis=(1, 2))[:, None, None]
        if node == self.top:
            return post.sum(axis=2)
        if node in self.middle:
            j = self.middle.index(node)
            q = post.sum(axis=1)
            return np.stack(
                [
                    q[:, self.combos[:, j] == v].sum(axis=1)
                    for v in range(net.cards[node])
                ],
                axis=1,
            )
        if node in self.leaves:
            full, obs, seen = leaf_tabs[node]
            out = np.einsum("nc,ncv->nv", post.sum(axis=1), full)
            out[seen] = 0.0
            out[seen, obs[seen].astype(int)] = 1.0
            return out
        raise KeyError(f"{self.name}: {node} is not a modeled node")
