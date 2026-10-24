# models/base.py
# modules/models/base.py
from abc import ABC, abstractmethod
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from cse250a.modules.eval.metrics import score


class Model(ABC):
    """Base for all models. Models receive already-binned DataFrames and have no
    knowledge of config, files, or splits."""

    name: ClassVar[str]

    def __init__(self, target: str):
        self.target = target

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Self:
        """Learn parameters from binned training data."""

    @abstractmethod
    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """P(target | evidence) per hour: one row per hour, one column per bin."""

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Most likely bin per hour."""
        return self.predict_proba(data).idxmax(axis=1)

    def evaluate(self, data: pd.DataFrame) -> dict:
        """Score on hours where the target is observed. Same metrics for every model."""
        observed = data[data[self.target].notna()]
        return score(observed[self.target], self.predict_proba(observed))