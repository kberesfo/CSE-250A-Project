# modules/core/results.py
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FitResult:
    """What fitting produced, kept for the report: convergence and stability.

    MLE: one pass, so n_iter = 1, converged = True, history = [final log-likelihood].
    EM: one entry per iteration. With alpha > 0 the entry is log-likelihood + log-prior
    (MAP-EM), which is the quantity EM never decreases.
    """

    history: list[float] = field(default_factory=list)  # objective per iteration
    n_iter: int = 0
    converged: bool = False

    @property
    def final(self) -> float:
        """Last recorded objective. NaN if nothing was recorded."""
        raise NotImplementedError

    def record(self, objective: float) -> None:
        """Append one iteration's objective and increment n_iter."""
        raise NotImplementedError

    def should_stop(self, tol: float) -> bool:
        """True when the last gain is below tol × |previous objective|.
        False with fewer than two entries."""
        raise NotImplementedError

    def is_monotone(self, rtol: float = 1e-8) -> bool:
        """True if the objective never dropped by more than rtol × |value|.
        A drop means a bug in the E-step or M-step, not bad luck."""
        raise NotImplementedError

    def to_frame(self) -> pd.DataFrame:
        """One row per iteration (columns: iteration, objective), for the convergence plot."""
        raise NotImplementedError