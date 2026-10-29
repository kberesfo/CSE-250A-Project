# modules/models/base.py
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from cse250a.modules.eval import score


class Model(ABC):
    """Base for all models. Models receive already-binned DataFrames and have no
    knowledge of config, files, or splits.

    Subclasses implement `fit` and `_predict_proba`. The public `predict_proba` is
    fixed here so every model hides the same columns before predicting: anything
    not in `evidence` (including the target) is set to NaN first. A model therefore
    cannot accidentally read the reference reading it is being scored against.
    """

    name: ClassVar[str] = "model"

    def __init__(self, target: str, evidence: Sequence[str] | None = None):
        self.target = target
        # Columns visible at predict time. None = no masking (only for baselines
        # that are deliberately given privileged information, e.g. persistence).
        self.evidence = tuple(evidence) if evidence is not None else None

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Self:
        """Learn parameters from binned training data."""

    @abstractmethod
    def _predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """P(target | evidence) per row, given data that has already been masked."""

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """P(target | evidence) per hour: one row per hour (same index as `data`),
        one column per bin. Do not override; override `_predict_proba`."""
        return self._predict_proba(self.mask(data))

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Most likely bin per hour."""
        return self.predict_proba(data).idxmax(axis=1)

    def mask(self, data: pd.DataFrame) -> pd.DataFrame:
        """Copy of `data` with every column outside `evidence` set to NaN.
        Rows are never dropped, so time-ordered models still see whole sequences."""
        if self.evidence is None:
            return data
        masked = data.copy()
        hidden = [c for c in masked.columns if c not in self.evidence]

        masked[hidden] = np.nan

        return masked

    def evaluate(self, data: pd.DataFrame) -> dict:
        """Score on hours where the target is observed. Same metrics for every model.

        Predicts on ALL rows first, then selects the observed ones. Filtering before
        predicting would cut holes in the hour sequence and break the HMM phases.
        """
        proba = self.predict_proba(data)
        observed = data[self.target].notna()
        return score(data.loc[observed, self.target], proba.loc[observed])
