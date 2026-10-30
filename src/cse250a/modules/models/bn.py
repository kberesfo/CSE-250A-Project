# modules/models/bn.py
import itertools
from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np
import pandas as pd

from cse250a.modules.core import CPT, FitResult

from .base import Model


class DiscreteBN(Model):
    """
    Discrete Bayesian network over binned columns, learned by EM. One class for
    every phase.

    The structure is passed in, so each phase is a dict built in its notebook:

        structure = {"CO(GT)": ["hour"], **{s: ["CO(GT)", "T", "RH"] for s in SENSORS}}
        model = DiscreteBN(structure, "CO(GT)", evidence=["hour", "T", "RH", *SENSORS],
                           cards={**disc.cards, **TIME_CARDS}).fit(train)

    Node kinds
      modeled   a key of `structure`; gets a CPT
      input     appears only as a parent (hour, weather); conditioned on, no CPT
      latent    modeled but never a data column (Phase 3b's Event); declared in
                `latent` so a misspelled column raises instead of becoming hidden

    Learning is EM over every row. Missing cells (a gas reference that is NaN)
    and latent nodes are both just hidden values; a latent node is hidden on every
    row. With no missing data and no latent nodes, EM stops after one step at the
    plain counting (MLE) answer, so Phases 1-2 cost nothing extra.

    Extension point: Phase 4 (HMM) overrides `_joint` / `posterior` with
    forward-backward; fit, the M-step and prediction stay as they are.
    """

    name = "discrete_bn"

    def __init__(
        self,
        structure: Mapping[str, Sequence[str]],
        target: str,
        *,
        evidence: Sequence[str],
        cards: Mapping[str, int],
        latent: Sequence[str] = (),
        alpha: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-4,
        seed: int = 0,
        init: Mapping[str, np.ndarray] | None = None,
    ):
        """
        Args:
            structure: Node → its parents. Every key gets a CPT.
            target: The node predict_proba and evaluate are about.
            evidence: Columns visible at predict time (hour, weather, sensors);
                never a reference reading.
            cards: Number of states for every node and input, e.g.
                {**discretizer.cards, **TIME_CARDS, "Event": 3}. Passed in, not
                inferred, so a bin missing from training still gets a column.
            latent: Nodes with no data column (e.g. ("Event",)). Their number of
                states must be in `cards`.
            alpha: Pseudo-count added to every CPT cell in each M-step.
            max_iter: EM iterations at most.
            tol: Stop when an iteration improves the objective by less than
                tol × |objective|.
            seed: For the random starting CPTs of latent nodes.
            init: Optional starting tables for chosen nodes, overriding the
                default start.
        """
        super().__init__(target, evidence)
        self.structure = {node: tuple(ps) for node, ps in structure.items()}
        self.cards = dict(cards)
        self.latent = tuple(latent)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.init = dict(init or {})
        self.cpts: dict[str, CPT] = {}
        self.result: FitResult | None = None  # EM history, for the convergence plot

    # ── learning ───────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> Self:
        """EM.

        1. Check every non-latent node and every input is a column of `train`
           (raise ValueError naming the missing ones), and that every node and
           input has an entry in `cards`.
        2. Add each latent node to `train` as an all-NaN column.
        3. self.cpts = self._init_cpts(train); self.result = FitResult().
        4. Repeat up to max_iter times:
             E-step:  counts, loglik = self._expected_counts(train)
             record:  self.result.record(loglik + self._log_prior())
             stop:    if self.result.should_stop(self.tol): converged, break
             M-step:  self.cpts[node] = CPT.from_counts(node, parents,
                                                         counts[node], self.alpha)
        With alpha > 0 this is MAP-EM: loglik + log-prior is the quantity that
        never decreases. self.result.is_monotone() checks that.
        """
        raise NotImplementedError

    def _init_cpts(self, train: pd.DataFrame) -> dict[str, CPT]:
        """Starting CPTs.

        Nodes whose family (node + parents) has no latent member: from complete-
        case counts (self._complete_counts), which is already the MLE when nothing
        is missing. Families touching a latent node: random rows from
        rng.dirichlet (rng = np.random.default_rng(self.seed)); random, not
        uniform, or the latent states never separate. Then self.init[node]
        replaces any node given there.
        """
        rng = np.random.default_rng(self.seed)
        complete = self._complete_counts(train)
        out = {}

        for node, parents in self.structure.items():
            cols = [*parents, node]
            shape = tuple(self.cards[c] for c in cols)

            if any(c in self.latent for c in cols):
                k = shape[-1]
                n_rows = int(np.prod(shape[:-1]))
                table = rng.dirichlet(np.ones(k), size=n_rows).reshape(shape)
                cpt = CPT(node, tuple(parents), table)
            else:
                cpt = CPT.from_counts(node, parents, complete[node], self.alpha)  # 4

            if node in self.init:
                cpt = CPT(
                    node, tuple(parents), np.asarray(self.init[node], dtype=float)
                )

            out[node] = cpt

        return out

    def _complete_counts(self, data: pd.DataFrame) -> dict[str, np.ndarray]:
        """Counts for every node: {node: array of shape (*parent cards, node card)},
        using only rows where the node and every parent are observed. Same idea as
        HourOnlyBaseline.fit, once per node. Families with a latent member get
        all zeros (no complete rows)."""
        out = {}
        for node, parents in self.structure.items():
            cols = [*parents, node]
            # shape from self.cards
            shape = tuple(self.cards[c] for c in cols)

            # if any column in cols is missing from data (a latent node):
            if any(c not in data.columns for c in cols):
                out[node] = np.zeros(shape)
                continue

            rows = data[cols].to_numpy()
            complete = ~np.isnan(rows).any(axis=1)
            counts = np.zeros(shape)
            keep_rows = rows[complete].astype(int)
            np.add.at(counts, tuple(keep_rows.T), 1)
            out[node] = counts

        return out

    def _expected_counts(
        self, data: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], float]:
        """E-step at the current CPTs.

        For every row and every node, add P(node, parents | row) into
        counts[node]: take (hidden, joint) from self._joint(row, keep_all=True),
        sum `joint` down to the family's hidden members, and use the observed
        values for the rest. With nothing hidden this adds 1 to a single cell,
        i.e. the complete-case counts.

        Returns (counts, loglik), where loglik = Σ_rows log P(row's observed
        cells), the normalizer of each row's joint before normalizing.
        """
        raise NotImplementedError

    def _log_prior(self) -> float:
        """Σ alpha · log θ over every CPT entry: the Dirichlet log-prior (up to a
        constant) that MAP-EM adds to the log-likelihood. 0 when alpha == 0."""
        raise NotImplementedError

    # ── inference ──────────────────────────────────────────────────────────

    def _predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """self.posterior(data, self.target) as a DataFrame: index = data.index,
        one column per target bin (0 … cards[target] - 1). Latent nodes are added
        as all-NaN columns first, as in fit."""
        raise NotImplementedError

    def _joint(self, row: pd.Series) -> tuple[list[str], np.ndarray]:
        """
        The posterior over the unobserved nodes of one row.

        Returns (hidden, joint): `hidden` lists the unobserved nodes (missing or
        latent), and joint[i, j, ...] = P(hidden = (i, j, ...) | the row's
        observed cells), so joint.sum() == 1. Built by multiplying every CPT's
        entry for each assignment of `hidden`, then normalizing.

        keep_all=False may leave out a missing node with nothing observed below
        it (its CPT sums to 1, so it can't change the answer); that keeps
        prediction fast. keep_all=True keeps every unobserved node, which the
        E-step needs. Rows with a missing input (hour, weather) can't be scored;
        drop them in preprocessing.

        Args:
            row (pd.Series): _description_
            keep_all (bool, optional): _description_. Defaults to False.

        Returns:
            tuple[list[str], np.ndarray]: _description_
        """
        # NOTE:
        hidden = [n for n in self.structure if n not in row.index or pd.isna(row[n])]
        joint = np.zeros(tuple(self.cards[h] for h in hidden))  # shape (3, 2)

        # print(list(zip(hidden, states)))  # [("G", 2), ("A", 1)]   pairs them up
        for states in itertools.product(*[range(self.cards[h]) for h in hidden]):
            # print(dict(zip(hidden, states)))  # {"G": 2, "A": 1}       look up by name
            assignment = dict(zip(hidden, states))
            factor = 1.0
            for node, parents in self.structure.items():
                keys = [*parents, node]
                # assignment[var] if var is hidden, else int(row[var])
                index = tuple(
                    assignment[var] if var in hidden else int(row[var]) for var in keys
                )
                factor *= self.cpts[node].table[index]
            joint[states] = factor

        joint = joint / joint.sum()

        return hidden, joint

    def posterior(self, data: pd.DataFrame, node: str) -> np.ndarray:
        """P(node | the observed cells of each row): shape (n_rows, cards[node]).
        Per row: sum `_joint` down to `node`, or a one-hot if `node` is observed.
        Works for latent nodes too, e.g. P(Event | row) for the rush-hour plot."""
        out = np.zeros((len(data), self.cards[node]))

        for i, (_, row) in enumerate(data.iterrows()):
            hidden, joint = self._joint(row)
            if node in hidden:
                j = hidden.index(node)
                # every axis number except j
                others = tuple([axis for axis in range(joint.ndim) if axis != j])

                out[i] = joint.sum(axis=others)
            else:
                out[i, int(row[node])] = 1

        return out

    def log_likelihood(self, data: pd.DataFrame) -> float:
        """Σ_rows log P(observed cells of the row). Held-out fit, and the
        quantity EM tracks."""
        

    # ── diagnostics ────────────────────────────────────────────────────────

    def sparsity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Per CPT: parent combinations, complete-case rows used, and how many
        combinations are empty or thin (< 10 rows). Check before trusting a
        structure. Families with a latent member are left out (EM fills them)."""
        raise NotImplementedError
