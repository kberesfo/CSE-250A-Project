# models/base.py
from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np
import pandas as pd


class Model(ABC):
    name: ClassVar[str]

    @abstractmethod
    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> Self:
        """Learn parameters from discretized training data."""

    @abstractmethod
    def predict_proba(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """P(target | evidence) per hour: one row per hour, one column per bin."""

    # concrete: every model gets these for free
    def predict(self, data, target) -> pd.Series:
        return self.predict_proba(data, target).idxmax(axis=1)

    def log_likelihood(self, data, target) -> float:
        p = self.predict_proba(data, target)
        y = data[target]
        mask = y.notna()
        return float(np.log(p[mask].to_numpy()[np.arange(mask.sum()), y[mask].astype(int)]).sum())