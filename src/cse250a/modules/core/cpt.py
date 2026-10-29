from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CPT:
    """P(node | parents) as an array with axes (*parents, node).

    table[p1, p2, ..., k] = P(node = k | parents = (p1, p2, ...)). Every slice
    along the last axis sums to 1. A counts table has the same shape, so the
    M-step is the same for MLE (hard counts) and EM (expected counts).
    """

    node: str
    parents: tuple[str, ...]
    table: np.ndarray

    @classmethod
    def from_counts(
        cls, node: str, parents: Sequence[str], counts: np.ndarray, alpha: float
    ) -> "CPT":
        """M-step: add `alpha` to every cell, normalize over the last axis.
        A parent combination with no counts (only possible when alpha = 0) becomes
        uniform instead of NaN."""
        raise NotImplementedError

    def lookup(self, data: pd.DataFrame) -> np.ndarray:
        """P(node value | parent values) for each row of `data`: shape (n_rows,).
        NaN on rows where the node or any parent is missing."""
        raise NotImplementedError

    def to_frame(self) -> pd.DataFrame:
        """Readable version for notebooks: one row per parent combination, one column per node state."""
        raise NotImplementedError
