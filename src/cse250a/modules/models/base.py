# modules/models/base.py
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
import pandas as pd

from cse250a.modules.core import Gas
from cse250a.modules.eval import score


class Model(ABC):
    """Base for all models. Models receive already-binned DataFrames and have no
    knowledge of config, files, or splits.

    A query is (target, evidence): "how well can we predict `target` from the
    columns in `evidence`?" Both are passed at call time, not stored on the
    model, so one fitted model answers many queries:

        bn = DiscreteBN(network).fit(train)
        bn.evaluate(test, Gas.CO, evidence)
        bn.evaluate(test, Gas.NO2, evidence)

    Subclasses implement `fit` and `_predict_proba`. The public `predict_proba`
    is fixed here so every model masks the same way before predicting: every
    column outside `evidence` is set to NaN, and `evidence` may never contain a
    reference reading (any Gas column) or the target itself. A model therefore
    cannot read the reference it is being scored against.

    Class attributes a subclass may set:
        name: Label for results tables and error messages.
        requires: Columns the model cannot predict without (e.g. hour for the
            hour baseline, every input for a BN). A query whose evidence lacks
            one raises. A subclass may set it per instance in __init__.
        privileged: True only for baselines that deliberately read the target's
            own history (persistence). Skips masking and the leak check; report
            such models as upper-information references, not competitors.
    """

    name: ClassVar[str] = "model"
    requires: ClassVar[tuple[str, ...]] = ()
    privileged: ClassVar[bool] = False

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> Self:
        """Learn parameters from binned training data.

        Args:
            train: Discretized training split (Discretizer.transform output).
                Missing cells are NaN.

        Returns:
            Self, so calls chain: model.fit(train).evaluate(test, target, evidence).
        """

    @abstractmethod
    def _predict_proba(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """P(target | evidence) per row. Implement this, not `predict_proba`.

        Args:
            data: Already masked by `mask`: every column outside the query's
                evidence is NaN, so treat NaN as unobserved.
            target: The column or node to predict.

        Returns:
            One row per row of `data` (same index), one column per target bin
            (0 … k-1); each row sums to 1.
        """

    def predict_proba(
        self, data: pd.DataFrame, target: str, evidence: Sequence[str]
    ) -> pd.DataFrame:
        """P(target | evidence) per hour. Masks `data`, then calls `_predict_proba`.
        Do not override; override `_predict_proba`.

        Args:
            data: Discretized rows. Rows are never dropped, so time-ordered
                models see whole sequences.
            target: The column or node to predict, e.g. Gas.CO or Gas.CO.latent.
            evidence: Columns the model may read. Everything else is hidden.

        Returns:
            One row per row of `data` (same index), one column per target bin.

        Raises:
            ValueError: From `mask`, if the evidence leaks a reference or lacks
                a required column.
        """
        return self._predict_proba(self.mask(data, target, evidence), target)

    def predict(
        self, data: pd.DataFrame, target: str, evidence: Sequence[str]
    ) -> pd.Series:
        """Most likely target bin per hour (the argmax of `predict_proba`).
        Arguments as in `predict_proba`."""
        return self.predict_proba(data, target, evidence).idxmax(axis=1)

    def mask(
        self, data: pd.DataFrame, target: str, evidence: Sequence[str]
    ) -> pd.DataFrame:
        """Copy of `data` with every column outside `evidence` set to NaN.

        Rows are never dropped, so time-ordered models still see whole sequences.
        For a privileged model, returns `data` unchanged (not a copy) after the
        `requires` check.

        Args:
            data: Discretized rows.
            target: The query's target; may not appear in `evidence`.
            evidence: Columns to keep visible.

        Returns:
            The masked copy.

        Raises:
            ValueError: If `evidence` is missing a column in `requires`, or
                (non-privileged models) contains a Gas reference column or the
                target.
        """
        evidence = tuple(evidence)

        missing = [c for c in self.requires if c not in evidence]
        if missing:
            raise ValueError(f"{self.name} needs {missing} in evidence")

        if self.privileged:
            return data

        leaked = (set(evidence) & set(Gas)) | ({target} & set(evidence))
        if leaked:
            raise ValueError(
                f"evidence includes reference columns {sorted(leaked)}; the score would leak"
            )

        masked = data.copy()
        hidden = [c for c in masked.columns if c not in evidence]
        masked[hidden] = np.nan

        return masked

    def evaluate(
        self,
        data: pd.DataFrame,
        target: str,
        evidence: Sequence[str],
        *,
        against: str | None = None,
    ) -> dict:
        """Score P(target | evidence) on hours where `against` is observed.
        Same metrics for every model (see eval.score).

        Predicts on ALL rows first, then selects the observed ones. Filtering
        before predicting would cut holes in the hour sequence and break the
        HMM phases.

        Args:
            data: Discretized rows, references included (they are hidden from
                the model by `mask`, and used here only for scoring).
            target: The column or node to predict.
            evidence: Columns the model may read.
            against: Column holding the true bins, defaulting to `target`.
                Differs only when the target is latent, e.g.
                evaluate(test, Gas.CO.latent, evidence, against=Gas.CO).

        Returns:
            score's dict: n, accuracy, macro_f1, bere, log_likelihood,
            mean_log_likelihood, confusion.
        """
        against = against or target
        proba = self.predict_proba(data, target, evidence)
        observed = data[against].notna()

        return score(data.loc[observed, against], proba.loc[observed])
