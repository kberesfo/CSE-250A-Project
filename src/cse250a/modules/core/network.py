# modules/core/network.py
"""What a network can be made of. Which network, which edges and which evidence
a phase uses is decided in that phase's notebook, not here."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Self

from .columns import Gas, Sensor, Time, Weather

# Every name a non-latent node or input may have: the dataset's columns.
COLUMNS: frozenset[str] = frozenset({*Gas, *Sensor, *Weather, *Time})

# Shapes the notebooks fill in.
Structure = Mapping[str, tuple[str, ...]]  # node → same-hour parents
Cards = Mapping[str, int]  # variable → number of states
Edges = Mapping[Sensor, tuple[Gas, ...]]  # sensor → gases it responds to
Evidence = tuple[str, ...]  # columns visible to a query


@dataclass(frozen=True)
class Network:
    """The structure of a discrete Bayesian network: the graph and the number of
    states of every variable. Nothing about queries: the target and the evidence
    are passed when predicting, so one fitted network answers many queries.

    Each phase builds its Network in its notebook, e.g.

        nominal = {Sensor.S1: (Gas.CO,), Sensor.S3: (Gas.NOX,), Sensor.S4: (Gas.NO2,)}
        network = Network(
            structure={
                **{g: (Time.HOUR,) for g in gases},
                **{s: (*nominal.get(s, ()), *weather) for s in Sensor},
            },
            cards={**disc.cards, **TIME_CARDS},
        ).validate(train.columns)

    Enum members are strings, so plain names ("CO(GT)", "hour") work as keys too.

    Node kinds
      modeled   a key of `structure`; gets a CPT
      input     appears only as a parent (hour, weather); conditioned on, no CPT,
                so it must be observed on every row and in every query's evidence
      latent    modeled, but never a data column (Phase 3's Gas.CO.latent == "Gas_CO")

    Treat a Network as read-only after validate(). (The dicts are plain copies,
    not MappingProxyType, so fitted models can be pickled.)
    """

    structure: Structure
    cards: Cards  # number of states for every modeled node and input
    latent: tuple[str, ...] = ()  # modeled nodes with no data column
    prev: Structure = field(default_factory=dict)
    # Phase 4: node → parents from the previous hour,
    # e.g. {Gas.CO.latent: (Gas.CO.latent,)}. Empty for a static network.

    def __post_init__(self):
        # copies, so the caller's dicts can't change the network afterwards
        def copy(m: Structure) -> dict[str, tuple[str, ...]]:
            return {k: tuple(v) for k, v in m.items()}

        object.__setattr__(self, "structure", copy(self.structure))
        object.__setattr__(self, "prev", copy(self.prev))
        object.__setattr__(self, "cards", dict(self.cards))
        object.__setattr__(self, "latent", tuple(self.latent))

    @property
    def nodes(self) -> tuple[str, ...]:
        """Modeled nodes in topological order (every parent before its children).
        Raises ValueError on a cycle within an hour."""
        return self._topo_order()

    @property
    def inputs(self) -> tuple[str, ...]:
        """Names that appear as a parent but are not keys of `structure`, in
        first-seen order."""
        seen = dict.fromkeys(p for ps in self.structure.values() for p in ps)
        return tuple(p for p in seen if p not in self.structure)

    @property
    def observed(self) -> tuple[str, ...]:
        """Modeled nodes that are data columns (nodes minus latent)."""
        return tuple(n for n in self.structure if n not in self.latent)

    @property
    def is_temporal(self) -> bool:
        """True when any node has parents in the previous hour."""
        return any(self.prev.values())

    def children(self, node: str) -> tuple[str, ...]:
        """Modeled nodes that list `node` as a same-hour parent."""
        return tuple(n for n, ps in self.structure.items() if node in ps)

    def family(self, node: str) -> tuple[str, ...]:
        """(*parents, node): the axes of the node's CPT, in order."""
        return (*self.structure[node], node)

    def shape(self, node: str) -> tuple[int, ...]:
        """CPT table shape for `node`: (*parent cards, node card), parents in
        `structure` order. For a temporal node, its previous-hour parents come first."""
        prev = tuple(self.cards[p] for p in self.prev.get(node, ()))
        return prev + tuple(self.cards[c] for c in self.family(node))

    def validate(self, columns: Iterable[str] | None = None) -> Self:
        """Raise ValueError listing every problem, else return self (for chaining).

        Checks:
          - every non-latent node and input is a known column (COLUMNS), so a
            typo raises instead of silently becoming a hidden node
          - every latent node is modeled and is not a known column
          - no cycle within an hour
          - every modeled node and input has a card ≥ 1
          - `prev` only names modeled nodes
        With `columns` (a DataFrame's columns), also: every observed node and
        input is a column of the data.
        """
        problems = []

        unknown = [c for c in (*self.observed, *self.inputs) if c not in COLUMNS]
        if unknown:
            problems.append(
                f"not dataset columns (typo, or forgot `latent`?): {unknown}"
            )

        stray = [z for z in self.latent if z not in self.structure]
        if stray:
            problems.append(f"latent nodes not in structure: {stray}")
        clash = [z for z in self.latent if z in COLUMNS]
        if clash:
            problems.append(f"latent nodes named like dataset columns: {clash}")

        try:
            self._topo_order()
        except ValueError as e:
            problems.append(str(e))

        no_card = [c for c in (*self.structure, *self.inputs) if c not in self.cards]
        if no_card:
            problems.append(f"no entry in cards for {no_card}")
        bad_card = [c for c, k in self.cards.items() if k < 1]
        if bad_card:
            problems.append(f"cards must be ≥ 1 for {bad_card}")

        for n, ps in self.prev.items():
            bad = [x for x in (n, *ps) if x not in self.structure]
            if bad:
                problems.append(f"prev[{n!s}] names unmodeled nodes {bad}")

        if columns is not None:
            cols = set(columns)
            absent = [c for c in (*self.observed, *self.inputs) if c not in cols]
            if absent:
                problems.append(f"columns missing from data: {absent}")

        if problems:
            raise ValueError("invalid network:\n  " + "\n  ".join(problems))
        return self

    def _topo_order(self) -> tuple[str, ...]:
        """Modeled nodes, parents first. Raises ValueError on a cycle within an hour."""
        order, done, visiting = [], set(), set()

        def visit(n: str) -> None:
            if n in done:
                return
            if n in visiting:
                raise ValueError(f"cycle within an hour through {n!s}")
            visiting.add(n)
            for p in self.structure.get(n, ()):
                if p in self.structure:
                    visit(p)
            visiting.discard(n)
            done.add(n)
            order.append(n)

        for n in self.structure:
            visit(n)
        return tuple(order)

    def __repr__(self) -> str:
        lines = [
            f"Network({len(self.structure)} nodes, inputs: {', '.join(map(str, self.inputs))})"
        ]
        for node in self.nodes:
            parents = ", ".join(map(str, self.structure[node]))
            kind = "  (latent)" if node in self.latent else ""
            lines.append(
                f"  {node!s:>14} | {parents:28}  CPT shape {self.shape(node)}{kind}"
            )
        return "\n".join(lines)
