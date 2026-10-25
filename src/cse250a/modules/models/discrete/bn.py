# modules/models/discrete/bn.py
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from ..base import Model


class DiscreteBN(Model):
    """Generic discrete Bayesian network over binned variables.

    Subclasses define STRUCTURE (node → list of parents). This class learns one CPT per node
    by MLE with Laplace smoothing and answers P(target | everything else observed).
    """

    STRUCTURE: ClassVar[dict[str, list[str]]] = {}

    def __init__(self, target: str, alpha: float = 1.0):
        super().__init__(target)
        self.alpha = alpha
        self.cpts: dict[str, pd.DataFrame] = {}  # node → P(node | parents)
        self.states: dict[str, list[int]] = {}  # node → its possible bin values
        self.rows_used: dict[str, int] = {}  # node → hours used to fit its CPT

    @property
    def structure(self) -> dict[str, list[str]]:
        """Node → parents. Override in subclasses that build the structure from constructor args."""
        return self.STRUCTURE

    def fit(self, train: pd.DataFrame) -> Self:
        """Record each node's states, then fit every CPT with _fit_cpt."""
        raise NotImplementedError

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """P(target | observed evidence) per row, normalized over the target's bins."""
        raise NotImplementedError

    def _fit_cpt(
        self, data: pd.DataFrame, node: str, parents: list[str]
    ) -> pd.DataFrame:
        """MLE counts on rows where node and all parents are observed, plus Laplace smoothing.

        Must include every parent combination, so unseen ones become uniform rows (not missing).
        """
        raise NotImplementedError

    def _log_factor(
        self, data: pd.DataFrame, node: str, target_value: int
    ) -> np.ndarray:
        """log P(node | parents) per row with the target set to target_value; 0 where node is missing."""
        raise NotImplementedError

    def sparsity(self) -> pd.DataFrame:
        """Per CPT: parent combinations, rows used, empty and thin (< 10 hours) combinations."""
        raise NotImplementedError
