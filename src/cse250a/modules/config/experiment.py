# modules/config/experiment.py
from enum import StrEnum
from itertools import pairwise
from typing import Literal

from pydantic import BaseModel, Field, model_validator


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
    """How continuous columns are binned. Mirrors Discretizer's arguments.

    Every continuous column gets `n_bins` quantile bins, unless overridden:
      per_column   a different bin count for some columns, e.g. {"CO(GT)": 5}
      fixed_edges  edges used as given instead of quantiles (health-standard
                   bands for AQ), e.g. {"NO2(GT)": [40, 90, 120]} → 4 bins

    Env vars use JSON for the dicts:
      BINS__N_BINS=5
      BINS__PER_COLUMN='{"CO(GT)": 5}'
      BINS__FIXED_EDGES='{"NO2(GT)": [40, 90, 120]}'
    """

    n_bins: int = Field(3, ge=2)
    per_column: dict[str, int] = Field(default_factory=dict)
    fixed_edges: dict[str, list[float]] = Field(default_factory=dict)
    # TODO: alpha is a model setting, not a binning one; move it if a ModelSettings group appears
    alpha: float = Field(1.0, ge=0)  # Laplace pseudo-count for every CPT

    @model_validator(mode="after")
    def _check(self):
        problems = []
        for col, k in self.per_column.items():
            if k < 2:
                problems.append(f"per_column[{col!r}] = {k}: need at least 2 bins")
        for col, edges in self.fixed_edges.items():
            if not edges:
                problems.append(f"fixed_edges[{col!r}] is empty")
            elif any(b <= a for a, b in pairwise(edges)):
                problems.append(
                    f"fixed_edges[{col!r}] must be strictly increasing: {edges}"
                )
        if both := set(self.per_column) & set(self.fixed_edges):
            problems.append(
                f"set either per_column or fixed_edges, not both: {sorted(both)}"
            )
        if problems:
            raise ValueError("; ".join(problems))
        return self
