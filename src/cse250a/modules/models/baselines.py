# modules/models/baselines.py
from typing import Self

import pandas as pd

from .base import Model


class HourOnlyBaseline(Model):
    """Predicts the target from hour of day alone: P(target | hour).

    The reference every phase must beat. Ignores sensors and weather.
    """

    name = "hour_baseline"

    def __init__(self, target: str, hour_col: str = "hour", alpha: float = 1.0):
        super().__init__(target)
        self.hour_col = hour_col
        self.alpha = alpha  # Laplace pseudo-count
        self.cpt: pd.DataFrame | None = None  # rows: hour (0–23), cols: target bins

    def fit(self, train: pd.DataFrame) -> Self:
        """Count target bins per hour on rows where the target is observed; smooth and normalize."""
        raise NotImplementedError

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Look up each row's hour in the CPT. One row per hour, one column per bin."""
        raise NotImplementedError


class PersistenceBaseline(Model):
    """Predicts the target from the previous hour's observed value: P(target_t | target_{t-1}).

    The reference for Phase 4 (HMM): temporal structure must beat "same as last hour."
    Rows whose previous hour is missing or in another block fall back to the hour-of-day prior.
    """

    name = "persistence_baseline"

    def __init__(self, target: str, hour_col: str = "hour", alpha: float = 1.0):
        super().__init__(target)
        self.hour_col = hour_col
        self.alpha = alpha
        self.transition: pd.DataFrame | None = (
            None  # rows: previous bin, cols: current bin
        )
        self.fallback = HourOnlyBaseline(target, hour_col, alpha)

    def fit(self, train: pd.DataFrame) -> Self:
        """Count (previous bin → current bin) pairs over consecutive hours; also fit the fallback."""
        raise NotImplementedError

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Use the transition row for the previous hour's bin, or the fallback if it's unavailable."""
        raise NotImplementedError

    def _previous(self, data: pd.DataFrame) -> pd.Series:
        """Target at t-1, NaN where the previous timestamp isn't exactly one hour earlier."""
        raise NotImplementedError
