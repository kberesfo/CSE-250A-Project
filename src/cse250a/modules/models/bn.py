# modules/models/discrete/bn.py
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd

from cse250a.modules.core.cpt import CPT

from .base import Model


class DiscreteBN(Model):
    """Discrete Bayesian network over binned columns. One class for every phase.

    The structure is passed in, not hard-coded, so each phase is a dict built in
    its notebook:

        structure = {"CO(GT)": ["hour"], **{s: ["CO(GT)", "T", "RH"] for s in SENSORS}}
        model = DiscreteBN(structure, target="CO(GT)", evidence=["hour", "T", "RH", *SENSORS],
                           cards={...}).fit(train)

    Keys of `structure` are the modeled nodes (each gets a CPT). Anything that only
    appears as a parent (hour, weather) is an input: conditioned on, never modeled.

    Extension points for later phases. Override these; don't edit fit or predict:
      _counts      LatentBN (Phase 3) returns expected counts instead of hard counts.
      posterior    Phase 4 (HMM) replaces per-row enumeration with forward-backward.
    A latent node (Phase 3's Gas_g) is a key of `structure` with no column in the data.
    """

    name = "discrete_bn"

    def __init__(
        self,
        structure: dict[str, list[str]],
        target: str,
        evidence: Sequence[str],
        cards: dict[str, int],
        alpha: float = 1.0,
    ):
        """
        structure   node → list of parents
        target      the node predict_proba and evaluate are about
        evidence    columns visible at predict time (hour, weather, sensors; never a reference)
        cards       number of bins per node and input, e.g. from the config's n_bins (hour: 24).
                    Passed in, not inferred, so a bin missing from training still gets a column.
        alpha       Laplace pseudo-count
        """
        super().__init__(target, evidence)
        self.structure = structure
        self.cards = cards
        self.alpha = alpha
        self.cpts: dict[str, CPT] = {}

    def fit(self, train: pd.DataFrame) -> Self:
        """counts = self._counts(train), then for every node
        self.cpts[node] = CPT.from_counts(node, parents, counts[node], self.alpha)."""
        raise NotImplementedError

    def _predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """self.posterior(data, self.target) as a DataFrame: index = data.index,
        one column per target bin (0 … cards[target]-1)."""
        raise NotImplementedError

    def _counts(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Complete-case counts for every node: {node: array of shape (*parent cards,
        node card)}, counting only rows where the node and every parent are observed."""
        raise NotImplementedError

    def _joint(self, row: pd.Series) -> tuple[list[str], np.ndarray]:
        """The posterior over every unobserved node in one row.

        Returns (hidden, joint): `hidden` lists the unobserved nodes (missing or
        latent), and joint[i, j, ...] = P(hidden = (i, j, ...) | the row's observed
        cells), so joint.sum() == 1. Built by multiplying every CPT's entry for each
        assignment of `hidden`, then normalizing. A missing node with nothing observed
        below it can be left out: its CPT sums to 1.

        `posterior` and LatentBN's E-step both read from this, so it is written once.
        """
        raise NotImplementedError

    def posterior(self, data: pd.DataFrame, node: str) -> np.ndarray:
        """P(node | the observed cells of each row): shape (n_rows, cards[node]).
        Per row: sum `_joint` down to `node`, or a one-hot if `node` is observed."""
        raise NotImplementedError

    def log_likelihood(self, data: pd.DataFrame) -> float:
        """Σ_rows log P(observed cells of the row). Held-out fit, and EM's convergence check."""
        raise NotImplementedError

    def sparsity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Per CPT: parent combinations, rows used, and how many combinations are
        empty or thin (< 10 rows). Check before trusting a structure."""
        raise NotImplementedError
