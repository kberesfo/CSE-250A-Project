# modules/models/baselines.py
from typing import Self

import numpy as np
import pandas as pd

from cse250a.modules.core import CPT, TIME_CARDS, Time

from .base import Model


class HourOnlyBaseline(Model):
    """Predicts the target from hour of day alone: P(target | hour).

    The reference every phase must beat. Ignores sensors and weather.
    Evidence is the hour column only, so the target is hidden at predict time.
    """

    name = "hour_baseline"

    def __init__(
        self,
        target: str,
        *,
        n_bins: int | None = None,
        alpha: float = 1.0,
        hour_col: str = Time.HOUR,
    ):
        super().__init__(target, evidence=[hour_col])
        self.hour_col = hour_col
        self.n_bins = n_bins  # target's bins; None = infer in fit (max bin + 1)
        self.alpha = alpha  # Laplace pseudo-count

        # P(target | hour): rows hour 0–23, cols target bins
        self.cpt: CPT | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        """
        Learn P(target | hour) by counting, then smoothing and normalizing.

        Uses only rows where both the hour and the target are observed (rows with a
        missing target carry no information about P(target | hour), so EM would give
        the same answer). Counts each (hour, target) pair, then builds the CPT with
        CPT.from_counts, which adds `alpha` to every cell and normalizes per hour.

        Args:
            train (pd.DataFrame): Discretized training data, i.e. the output of
                Discretizer.transform, not raw concentrations. Must contain the hour
                column (whole numbers 0-23) and the target column (bin indices
                0 to n_bins - 1, or NaN where missing). Other columns are ignored.

        Returns:
            Self: The fitted model, so calls can chain: model.fit(train).evaluate(test).

        Raises:
            ValueError: If no row has both the hour and the target observed (only
                when n_bins is None, since then the number of bins can't be inferred).
            ValueError: If the hour or target values are not whole numbers in range,
                which usually means the data was not discretized first.
        """
        addresses = train[[self.hour_col, self.target]].to_numpy()
        complete = ~np.isnan(addresses).any(axis=1)
        values = addresses[complete]  # still floats: check before casting
        n_hours = TIME_CARDS[Time.HOUR]
        # AI USED FOR DEFENSIVE PROGRAMMING
        if len(values) == 0 and self.n_bins is None:
            raise ValueError(
                f"no rows with both {self.hour_col!r} and {self.target!r} observed; "
                "pass n_bins or check the data"
            )
        if np.any(values != np.floor(values)):
            raise ValueError(
                f"{self.hour_col!r} and {self.target!r} must hold whole-number bins; "
                "found fractional values. Did you run the Discretizer?"
            )
        if len(values):
            hours, ys = values[:, 0], values[:, 1]
            if hours.min() < 0 or hours.max() >= n_hours:
                raise ValueError(
                    f"{self.hour_col!r} must be in 0..{n_hours - 1}, "
                    f"got {hours.min():g}..{hours.max():g}"
                )
            too_big = self.n_bins is not None and ys.max() >= self.n_bins
            if ys.min() < 0 or too_big:
                expected = "≥ 0" if self.n_bins is None else f"in 0..{self.n_bins - 1}"
                raise ValueError(
                    f"{self.target!r} bins must be {expected}, "
                    f"got {ys.min():g}..{ys.max():g}. Did you run the Discretizer?"
                )
        # AI ENDS
        pairs = values.astype(int)

        k = self.n_bins if self.n_bins is not None else int(pairs[:, 1].max()) + 1
        counts = np.zeros((n_hours, k))
        np.add.at(counts, tuple(pairs.T), 1)

        self.cpt = CPT.from_counts(self.target, [self.hour_col], counts, self.alpha)

        return self

    def _predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Look up each row's hour in the CPT. One row per hour, one column per bin."""
        if self.cpt is None:
            raise RuntimeError(f"{self.name}: call fit() before predicting")
        
        hours = data[self.hour_col].to_numpy().astype(int)
        rows = self.cpt.table[hours]
        k = self.cpt.table.shape[-1]

        return pd.DataFrame(rows, index=data.index, columns=range(k))


class PersistenceBaseline(Model):
    """Predicts the target from the previous hour's observed value: P(target_t | target_{t-1}).

    The reference for Phase 4 (HMM): temporal structure must beat "same as last hour."
    Rows whose previous hour is missing or in another block fall back to the hour-of-day prior.

    PRIVILEGED: this baseline reads the reference at t-1, which no phase model gets
    (their evidence is hour, sensors and weather). It is not masked (evidence=None),
    so `_predict_proba` must only ever read the target through `_previous`. Report it
    as an upper-information reference, not a like-for-like competitor.
    """

    name = "persistence_baseline"

    def __init__(
        self,
        target: str,
        *,
        n_bins: int | None = None,
        alpha: float = 1.0,
        hour_col: str = Time.HOUR,
    ):
        super().__init__(target, evidence=None)  # privileged: reads the target at t-1
        self.hour_col = hour_col
        self.n_bins = n_bins
        self.alpha = alpha
        self.transition: CPT | None = None  # P(target_t | target_{t-1})
        self.fallback = HourOnlyBaseline(
            target, n_bins=n_bins, alpha=alpha, hour_col=hour_col
        )

    def fit(self, train: pd.DataFrame) -> Self:
        """Count (previous bin → current bin) pairs over consecutive hours; also fit the fallback."""
        raise NotImplementedError

    def _predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Look up each row's hour in the CPT: P(target | hour) for every bin.
        One row per data row, one column per target bin."""
        raise NotImplementedError

    def _previous(self, data: pd.DataFrame) -> pd.Series:
        """Target at t-1, NaN where the previous timestamp isn't exactly one hour earlier."""
        raise NotImplementedError
