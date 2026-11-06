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
        """
        Build a CPT from counts: the M-step shared by MLE and EM.

        Adds `alpha` to every cell, then normalizes each parent combination so it sums
        to 1 over the node's states:

            P(node = j | parents) = (n_j + alpha) / (N + k * alpha)

        where n_j is the count for node state j, N is the total count for that parent
        combination, and k is the number of node states. A parent combination with no
        counts at all (only possible when alpha = 0) becomes uniform (1/k) instead of NaN.

        Args:
            node (str): Name of the node this CPT describes.
            parents (Sequence[str]): Names of the node's parents, in the same order as
                the leading axes of `counts`.
            counts (ArrayLike): Counts with shape (*parent cards, node card): one axis
                per parent, then one for the node. counts[p1, ..., pm, j] is how many
                rows had parents = (p1, ..., pm) and node = j. Hard counts for MLE,
                expected (fractional) counts for EM.
            alpha (float): Pseudo-count added to every cell (Laplace smoothing). It
                acts as if each node state was seen alpha extra times for every parent
                combination. 0 gives the plain MLE.

        Raises:
            ValueError: If counts doesn't have len(parents) + 1 axes.

        Returns:
            CPT: table[p1, ..., pm, j] = P(node = j | parents = (p1, ..., pm)).

        Example:
            counts = [[3, 1],      alpha = 1   →   table = [[4/6, 2/6],
                      [0, 0]]                               [1/2, 1/2]]
            (parent state 1 was never seen, so its row is uniform)
        """
        # accept lists too; always a plain float array
        counts = np.asarray(counts, dtype=float)
        # one axis per parent, then one for the node
        if counts.ndim != len(parents) + 1:
            raise ValueError(
                f"CPT {node!r}: counts has {counts.ndim} axes, expected "
                f"{len(parents) + 1} (one per parent plus the node)"
            )

        # add alpha to every cell
        smoothed = counts + alpha
        # sum across each row: one total per parent combination
        totals = smoothed.sum(axis=-1, keepdims=True)
        # number of node states (bins)
        k = counts.shape[-1]
        # uniform by default, for parent combinations with no counts
        result = np.full_like(smoothed, 1.0 / k)
        # divide each row by its total, only where the total is nonzero
        np.divide(smoothed, totals, out=result, where=totals > 0)

        return cls(node, tuple(parents), result)

    def lookup(self, data: pd.DataFrame) -> np.ndarray:
        """
        Probability the CPT assigns to each row's observed values.

        For each row, reads the cell addressed by its parent values and its node value:

            out[i] = table[p1_i, ..., pm_i, x_i] = P(node = x_i | parents = (p1_i, ..., pm_i))

        Rows where the node or any parent is missing get NaN; there is no single cell to
        read. (Summing over a missing value is inference, not lookup.)

        Args:
            data (pd.DataFrame): Binned data. Must contain a column for the node and each
                parent, holding bin indices (0 to card - 1) or NaN. Other columns are
                ignored, and column order doesn't matter.

        Returns:
            np.ndarray: Shape (n_rows,), aligned with the rows of `data`: one probability
                per row, NaN where the row is incomplete.
        """
        # each row's address into the table: parent values, then the node's value
        addresses = data[[*self.parents, self.node]].to_numpy()
        # a row can be looked up only if every address value is present
        complete = ~np.isnan(addresses).any(axis=1)

        out = np.full(len(data), np.nan)
        # one index array per table axis, paired row by row
        axes = addresses[complete].astype(int).T
        out[complete] = self.table[tuple(axes)]
        return out

    def to_frame(self) -> pd.DataFrame:
        """Readable view of the table for notebooks and the report.

        Flattens the table to 2-D: one row per parent combination, one column per node
        state. Each row is P(node | that parent combination) and sums to 1.

            rows      MultiIndex over the parents, one level per parent, named after it.
                      Rows run in table order, last parent changing fastest:
                      (0, 0), (0, 1), ..., (1, 0), ...
            columns   the node's states 0 ... k-1

        A root node (no parents) gives a single row.

        Returns:
            pd.DataFrame: Shape (product of parent cards, node card).

        Example:
            P(S1 | CO, T) with CO: 3 bins, T: 2 bins, S1: 4 bins → shape (6, 4)

                            0     1     2     3
                CO(GT) T
                0      0  0.17  0.19  0.29  0.35
                       1  0.05  0.48  0.28  0.20
                ...
        """
        # parent cardinalities, then the node's number of states
        *parent_cards, k = self.table.shape
        # flatten to (parent combinations, node states)
        flat = self.table.reshape(-1, k)
        if parent_cards:
            # row labels: every parent-value combination, last parent fastest
            # (the same order reshape uses)
            index = pd.MultiIndex.from_product(
                [range(card) for card in parent_cards], names=list(self.parents)
            )
        else:
            # root node: one row, P(node) itself
            index = pd.RangeIndex(1)
        # rows labeled by parent combination, columns by node state
        return pd.DataFrame(flat, index=index, columns=range(k))
