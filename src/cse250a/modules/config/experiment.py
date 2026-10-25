# modules/config/experiment.py
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, model_validator


class BinMethod(StrEnum):
    QUANTILE = "quantile"
    FIXED = "fixed"  # health-band edges, if you keep that comparison


class SplitMethod(StrEnum):
    TIME = "time"
    BLOCKED = "blocked"


class SplitSettings(BaseModel):
    train: float = 0.50
    val: float = 0.25
    test: float = 0.25
    method: Literal["time", "blocked"] = "time"
    block: str = "7D"  # block length for method="blocked"

    @model_validator(mode="after")
    def _sums_to_one(self):
        if abs(self.train + self.val + self.test - 1) > 1e-9:
            raise ValueError("train + val + test must equal 1")
        return self


class BinSettings(BaseModel):
    n_bins: int = 3
    method: BinMethod = BinMethod.QUANTILE
    alpha: float = 1.0
