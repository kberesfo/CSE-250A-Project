# modules/core/messages.py
from dataclasses import dataclass

import numpy as np


@dataclass
class Messages:
    """Forward-backward output for a run of hours (Phase 4).

    S is the number of joint states of the chain variables (e.g. 3**4 = 81 for four
    gases at 3 levels). Rows follow the data; each segment (a contiguous run of
    hours, see `starts`) is an independent chain.

    Uses scaled messages (Rabiner): alpha is normalized at every hour and `scale`
    stores the normalizers, so nothing underflows and log P(evidence) = Σ log scale.
    """

    alpha: np.ndarray  # (T, S) filtered: P(s_t | evidence up to t)
    beta: np.ndarray  # (T, S) scaled backward messages; 1 at each segment's last hour
    scale: np.ndarray  # (T,) c_t, the normalizer of alpha at hour t
    starts: np.ndarray  # (T,) bool, True at the first hour of each segment
    loglik: float  # log P(all evidence) = Σ_t log c_t (plus any per-hour offsets)

    @property
    def filtered(self) -> np.ndarray:
        """P(s_t | evidence up to t): (T, S). Deployable: uses no future hours."""
        raise NotImplementedError

    @property
    def smoothed(self) -> np.ndarray:
        """γ_t = P(s_t | all evidence in its segment): alpha · beta, renormalized per row. (T, S)."""
        raise NotImplementedError

    def pairwise(self, t: int, A: np.ndarray, e: np.ndarray) -> np.ndarray:
        """ξ_t(s', s) = P(s_{t-1} = s', s_t = s | all evidence): (S, S).

        A: (S, S) transition probabilities used at hour t.
        e: (S,) emission likelihoods at hour t (same scaling used in the forward pass).
        ξ_t = alpha[t-1][:, None] · A · (e · beta[t])[None, :] / scale[t].
        Undefined at a segment start (no previous hour): raise ValueError.
        """
        raise NotImplementedError

    def check(self) -> None:
        """Raise ValueError if shapes disagree or any alpha row doesn't sum to 1."""
        raise NotImplementedError
