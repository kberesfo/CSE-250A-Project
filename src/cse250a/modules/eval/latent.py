# modules/eval/latent.py
"""Checks for models with a latent node (Phase 3b's Event).

EM can stop at different local optima, and a latent state's label is arbitrary
(state 0 in one run can be state 2 in another). These helpers make runs
comparable:

    state_order        sort states by how much of a child they emit
    state_profile      a latent node's children's CPTs, one row per state
    restart_agreement  how far each restart is from the best one, after
                       matching state labels (Hungarian algorithm)
    free_params        number of free parameters of a network
"""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from cse250a.modules.core import Network


def free_params(network: Network) -> int:
    """Σ over nodes of (#parent configurations) × (card − 1)."""
    return int(
        sum(
            np.prod(network.shape(n)[:-1]) * (network.shape(n)[-1] - 1)
            for n in network.structure
        )
    )


def state_profile(bn, latent: str, children: Sequence[str]) -> np.ndarray:
    """Stack the children's CPTs into a (K, ...) array, one row per latent state.

    Each child's table must have the latent node as its first parent, e.g.
    P(gas | Event) or P(gas | Event, hour); any further parents are flattened.
    """
    k = bn.network.cards[latent]
    for c in children:
        if bn.network.structure[c][:1] != (latent,):
            raise ValueError(f"{c}: {latent} must be its first parent")
    return np.concatenate([bn.cpts[c].table.reshape(k, -1) for c in children], axis=1)


def state_order(bn, latent: str, by: str) -> np.ndarray:
    """State indices sorted by E[bin of `by` | state], lowest first.

    Used to give states stable names across seeds and restarts (e.g. by CO
    level: quiet → moderate → rush hour).
    """
    table = state_profile(bn, latent, [by]).reshape(
        bn.network.cards[latent], -1, bn.network.cards[by]
    )
    return np.argsort(table.mean(axis=1) @ np.arange(bn.network.cards[by]))


def restart_agreement(profiles: Sequence[np.ndarray]) -> list[float]:
    """Largest |ΔP| between each profile and the first, after matching states.

    `profiles[0]` is the reference (the best restart). States of every other
    profile are relabeled to best match it (minimum total absolute difference,
    Hungarian algorithm); the result is the largest remaining difference in
    any probability. Small values (< 0.1) mean the restarts found the same
    solution up to relabeling.
    """
    ref, out = profiles[0], []
    for p in profiles[1:]:
        cost = np.abs(ref[:, None, :] - p[None, :, :]).sum(axis=2)
        _, col = linear_sum_assignment(cost)
        out.append(float(np.abs(ref - p[col]).max()))
    return out
