# modules/models/baselines.py
from collections.abc import Mapping
from typing import Self

import numpy as np
import pandas as pd

from cse250a.modules.core import CPT, TIME_CARDS, Time

from .base import Model


class HourOnlyBaseline(Model):
    """Predicts the target from hour of day alone: P(target | hour).

    The reference every phase must beat. Whatever evidence a query passes, only
    the hour is read; sensors and weather are ignored. The hour must be in the
    evidence (`requires`). One fit covers every column in `cards`, so the target
    is chosen per query: model.evaluate(test, Gas.CO, evidence).
    """

    name = "hour_baseline"
    requires = (Time.HOUR,)
    
    def __init__(self, *, cards: Mapping[str, int], alpha: float = 1.0):
        """
        Args:
            cards: Number of bins per column, e.g. {**disc.cards, **TIME_CARDS}.
                The same dict the BN gets, so both agree on bin counts. Passed in,
                not inferred, so a bin missing from training still gets a column.
            alpha: Laplace pseudo-count added to every CPT cell.
        """
        self.cards = dict(cards)
        self.alpha = alpha

        # P(column | hour) per binned column: rows hour 0–23, cols target bins
        self.cpts: dict[str, CPT] = {}

    def fit(self, train: pd.DataFrame) -> Self:
        """
        Learn P(column | hour) for every binned column by counting, smoothing,
        normalizing. Binned columns are the keys of `cards` that are columns of
        `train`, minus the time columns (TIME_CARDS).

        Per column, uses only rows where both the hour and the target are observed
        (rows with a missing target carry no information about P(target | hour), so
        EM would give the same answer). Counts each (hour, target) pair, then builds
        the CPT with CPT.from_counts, which adds `alpha` to every cell and normalizes
        per hour. A target with no observed rows gets uniform rows (alpha only).

        Args:
            train (pd.DataFrame): Discretized training data, i.e. the output of
                Discretizer.transform, not raw concentrations. Must contain the hour
                column (whole numbers 0-23); binned columns hold bin indices
                0 to cards[col] - 1, or NaN where missing. Columns without an
                entry in `cards` are ignored.

        Returns:
            Self: The fitted model, so calls can chain:
                model.fit(train).evaluate(test, Gas.CO, evidence).

        Raises:
            ValueError: If the hour column is missing from `train`, no column
                of `cards` is in `train`, or the hour or bin values are not
                whole numbers in range, which usually means the data was not
                discretized first.
        """
        if Time.HOUR not in train.columns:
            raise ValueError(f"{self.name}: {Time.HOUR.value!r} missing from train")

        columns = [c for c in self.cards if c in train.columns and c not in TIME_CARDS]
        if not columns:
            raise ValueError(f"{self.name}: no column of cards is in train")

        self.cpts = {c: self._fit_one(train, c) for c in columns}

        return self

    def _fit_one(self, train: pd.DataFrame, target: str) -> CPT:
        """P(target | hour) for one target, from complete (hour, target) rows."""
        n_hours = TIME_CARDS[Time.HOUR]
        k = self.cards[target]

        addresses = train[[Time.HOUR, target]].to_numpy(dtype=float)
        complete = ~np.isnan(addresses).any(axis=1)
        values = addresses[complete]  # still floats: check before casting

        # AI USED FOR DEFENSIVE PROGRAMMING
        if np.any(values != np.floor(values)):
            raise ValueError(
                f"{Time.HOUR.value!r} and {target!r} must hold whole-number bins; "
                "found fractional values. Did you run the Discretizer?"
            )
        if len(values):
            hours, ys = values[:, 0], values[:, 1]
            if hours.min() < 0 or hours.max() >= n_hours:
                raise ValueError(
                    f"{Time.HOUR.value!r} must be in 0..{n_hours - 1}, "
                    f"got {hours.min():g}..{hours.max():g}"
                )
            if ys.min() < 0 or ys.max() >= k:
                raise ValueError(
                    f"{target!r} bins must be in 0..{k - 1}, "
                    f"got {ys.min():g}..{ys.max():g}. Did you run the Discretizer?"
                )
        # AI ENDS

        pairs = values.astype(int)
        counts = np.zeros((n_hours, k))
        np.add.at(counts, tuple(pairs.T), 1)

        return CPT.from_counts(target, [Time.HOUR], counts, self.alpha)

    def _predict_proba(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Look up each row's hour in the target's CPT. One row per data row,
        one column per target bin."""
        if not self.cpts:
            raise RuntimeError(f"{self.name}: call fit() before predicting")
        if target not in self.cpts:
            raise KeyError(
                f"{self.name}: not fitted for {target!r}; fitted {list(self.cpts)}"
            )

        hours = data[Time.HOUR].to_numpy(dtype=float)
        if np.isnan(hours).any():
            raise ValueError(f"{self.name}: {Time.HOUR.value!r} has missing values")

        cpt = self.cpts[target]
        rows = cpt.table[hours.astype(int)]
        k = cpt.table.shape[-1]

        return pd.DataFrame(rows, index=data.index, columns=range(k))


class PersistenceBaseline(Model):
    """
    Predicts the target from the previous hour's observed value:
    P(target_t | target_{t-1}).

    The reference for Phase 4 (HMM): temporal structure must beat "same as last hour."
    Rows whose previous hour is missing or in another block fall back to the
    hour-of-day prior.

    PRIVILEGED: this baseline reads the reference at t-1, which no phase model gets
    (their evidence is hour, sensors and weather). `privileged = True` makes
    Model.mask skip masking and the leak check, and hands over `data` itself (not
    a copy), so `_predict_proba` must only ever read the target through
    `_previous` and must not modify `data`. Report it as an upper-information
    reference, not a like-for-like competitor.
    """

    name = "persistence_baseline"
    requires = (Time.HOUR,)  # for the fallback
    privileged = True

    def __init__(self, *, cards: Mapping[str, int], alpha: float = 1.0):
        self.cards = dict(cards)
        self.alpha = alpha
        # P(col_t | col_{t-1}) per binned column, same columns as the fallback
        self.transitions: dict[str, CPT] = {}
        self.fallback = HourOnlyBaseline(cards=cards, alpha=alpha)

    def fit(self, train: pd.DataFrame) -> Self:
        """Per column, count (previous bin → current bin) pairs over consecutive
        hours; also fit the fallback."""
        raise NotImplementedError

    def _predict_proba(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """P(target_t | target_{t-1}) where the previous hour is observed, else the
        fallback's P(target | hour). One row per data row, one column per target
        bin. Call self.fallback._predict_proba(data, target) directly: `data` is
        already unmasked here."""
        raise NotImplementedError

    def _previous(self, data: pd.DataFrame, target: str) -> pd.Series:
        """`target` at t-1, NaN where the previous timestamp isn't exactly one hour
        earlier. Build it as a new Series (e.g. with .shift); don't add a column
        to `data`."""
        raise NotImplementedError