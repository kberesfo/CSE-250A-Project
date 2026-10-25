from typing import Self

import numpy as np
import pandas as pd


class Discretizer:
    """Quantile bin edges fit on training data, applied unchanged to any split."""

    def __init__(self, n_bins: int):
        self.n_bins = n_bins
        self.edges: dict[str, np.ndarray] = {}

    def fit(self, train: pd.DataFrame, cols: list[str]) -> Self:
        """Learn n_bins - 1 interior quantile edges per column, ignoring NaN."""
        raise NotImplementedError

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace each fitted column with its bin index (0 … n_bins-1); NaN stays NaN.
        Columns not fitted (e.g. hour) pass through unchanged.
        """
        raise NotImplementedError

    def fit_transform(self, train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return self.fit(train, cols).transform(train)
