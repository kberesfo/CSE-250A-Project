# modules/models/discrete/latent.py
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd

from .bn import CPT, DiscreteBN


class LatentBN(DiscreteBN):
    """DiscreteBN with latent nodes, learned by EM (Phase 3).

    Same structure dict as DiscreteBN. A latent node is a key with no column in the
    data, e.g. Phase 3 in its notebook:

        structure = {
            **{f"Gas_{g}": ["hour"] for g in GASES},          # latent true levels
            **{f"{g}(GT)": [f"Gas_{g}"] for g in GASES},      # references: noisy children
            "PT08.S1(CO)": ["Gas_CO", "T", "RH"], ...         # sensors hang off Gas_g
        }
        model = LatentBN(structure, target="CO(GT)", evidence=[...], cards={...},
                         latent=[f"Gas_{g}" for g in GASES],
                         init={f"{g}(GT)": anchor for g in GASES}).fit(train)

    where `anchor` is a (n_bins, n_bins) table with ~0.8 on the diagonal (rows sum to 1).

    Only `fit`, `_counts` and `_init_cpts` change. Prediction, posterior and
    evaluation are inherited: a latent node is just a node that is missing on every row.
    """

    name = "latent_bn"

    def __init__(
        self,
        structure: dict[str, list[str]],
        target: str,
        evidence: Sequence[str],
        cards: dict[str, int],
        latent: Sequence[str],
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        init: dict[str, np.ndarray] | None = None,
        seed: int = 0,
    ):
        """
        latent     nodes with no data column (their number of states goes in `cards`)
        max_iter   EM iterations at most
        tol        stop when the objective improves by less than tol × |objective|
        init       starting tables for chosen nodes, e.g. P(Ref_g | Gas_g) near the
                   identity so latent state k means bin k. Every other CPT starts random.
        seed       for the random starting CPTs
        """
        super().__init__(structure, target, evidence, cards, alpha)
        self.latent = list(latent)
        self.max_iter = max_iter
        self.tol = tol
        self.init = init or {}
        self.seed = seed
        self.history: list[
            float
        ] = []  # objective per iteration, for the convergence plot

    def fit(self, train: pd.DataFrame) -> Self:
        """EM.
        1. Add each latent node to `train` as an all-NaN column.
        2. self.cpts = self._init_cpts().
        3. Repeat up to max_iter times:
             E-step: counts = self._counts(train)          (expected counts at current CPTs)
             record objective = log_likelihood + Σ alpha · log θ over every CPT entry
             stop if it improved by less than tol × |previous objective|
             M-step: self.cpts[node] = CPT.from_counts(node, parents, counts[node], self.alpha)
        With alpha > 0 this is MAP-EM: the objective above (not the log-likelihood
        alone) is the quantity that never decreases. Plot self.history to check.
        """
        raise NotImplementedError

    def _init_cpts(self) -> dict[str, CPT]:
        """Starting CPTs: each one random (rng.dirichlet over the last axis, using
        self.seed), then replaced by self.init[node] where given. Random, not
        uniform: identical latent states would never separate."""
        raise NotImplementedError

    def _counts(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        """E-step: expected counts for every node, same shapes as DiscreteBN._counts.

        For each row, get (hidden, joint) from self._joint(row) and add
        P(node, parents | row) into counts[node]: sum `joint` down to the hidden
        members of the family, and use the observed values for the rest. With
        nothing missing this adds exactly 1 to one cell, i.e. the hard counts, so
        EM on complete data gives the same CPTs as DiscreteBN.
        """
        raise NotImplementedError
