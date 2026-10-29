# modules/models/baselines.py
from typing import Self

import pandas as pd

from cse250a.modules.core import CPT, HOUR

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
        hour_col: str = HOUR,
    ):
        super().__init__(target, evidence=[hour_col])
        self.hour_col = hour_col
        self.n_bins = n_bins  # target's bins; None = infer in fit (max bin + 1)
        self.alpha = alpha  # Laplace pseudo-count

        # P(target | hour): rows hour 0–23, cols target bins
        self.cpt: CPT | None = None

    def fit(self, train: pd.DataFrame) -> Self:
        """Count target bins per hour on rows where the target is observed; smooth and normalize."""
        raise NotImplementedError

    def _predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Look up each row's hour in the CPT. One row per hour, one column per bin."""
        raise NotImplementedError


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
        hour_col: str = HOUR,
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
