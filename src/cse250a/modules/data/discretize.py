from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import pandas as pd


class Discretizer:
    """Bin continuous columns into integers 0 … k-1.

    Edges are learned on training data only (quantiles), or given up front
    (fixed edges, e.g. health-standard bands for AQ), then applied unchanged
    to every split. Columns not fitted (hour, is_weekday, ...) pass through.
    """

    def __init__(
        self,
        n_bins: int = 3,
        *,
        per_column: Mapping[str, int] | None = None,
        fixed_edges: Mapping[str, Sequence[float]] | None = None,
    ):
        """
        Args:
            n_bins: Default number of bins for every quantile-binned column.
            per_column: Bin counts that override `n_bins` for specific columns,
                e.g. {"CO(GT)": 5}.
            fixed_edges: Interior edges to use as given instead of learning
                quantiles, e.g. {"NO2(GT)": [40, 90, 120]} → 4 bins. Must be
                increasing. Takes priority over `n_bins` / `per_column`.
        """
        self.n_bins = n_bins
        self.per_column = dict(per_column or {})
        self.fixed_edges = {
            c: np.asarray(e, dtype=float) for c, e in (fixed_edges or {}).items()
        }
        # learned (or fixed) interior edges per fitted column; k bins → k - 1 edges
        self.edges: dict[str, np.ndarray] = {}

    def bins_for(self, col: str) -> int:
        """Number of bins requested for `col` (quantile columns only)."""
        return self.per_column.get(col, self.n_bins)

    @property
    def cards(self) -> dict[str, int]:
        """Number of bins per fitted column: k - 1 edges → k bins.
        Pass to models as their `n_bins` / `cards`."""
        return {col: len(e) + 1 for col, e in self.edges.items()}

    def fit(self, train: pd.DataFrame, cols: Sequence[str]) -> Self:
        """Learn interior edges for each column in `cols`, ignoring NaN.

        Fixed-edge columns use their given edges. Every other column gets
        bins_for(col) - 1 quantile edges, evenly spaced in probability
        (for 3 bins: the 1/3 and 2/3 quantiles). Stores them in self.edges.
        """
        for col in cols:
            if col in self.fixed_edges:
                edges = self.fixed_edges.get(col)
                # check if each edge is bigger than the last
                if np.any(np.diff(edges) <= 0):
                    raise ValueError(
                        f"fixed edges for {col!r} must be strictly increasing, got {list(edges)}"
                    )
            else:
                k = self.bins_for(col)
                fractions = np.arange(1, k) / k
                edges = np.nanquantile(train[col].to_numpy(), fractions)
            self.edges[col] = edges

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Copy of `df` with each fitted column replaced by its bin index
        (0 … k-1); NaN stays NaN. Columns not fitted pass through unchanged."""
        out = df.copy()

        for col, edges in self.edges.items():
            x = df[col].to_numpy(dtype=float)
            bins = np.digitize(x, edges).astype(float)
            nan_loc = np.isnan(x)
            bins[nan_loc] = np.nan 
            out[col] = bins

        return out

    def fit_transform(self, train: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        return self.fit(train, cols).transform(train)
