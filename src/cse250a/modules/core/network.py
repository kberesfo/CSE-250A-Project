# modules/core/network.py
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Self

from .columns import HOUR, Gas, Sensor, Weather  # noqa: F401

SENSORS = tuple(Sensor)
GASES = (Gas.CO, Gas.C6H6, Gas.NOX, Gas.NO2)  # NMHC left out
WEATHER = (Weather.T, Weather.RH)  # docs/decisions.md 2026-10-19

# Solid edges: the gas each sensor targets. Ignored when that gas isn't in the model.
NOMINAL: dict[Sensor, tuple[Gas, ...]] = {
    Sensor.S1: (Gas.CO,),
    Sensor.S2: (Gas.NMHC,),
    Sensor.S3: (Gas.NOX,),
    Sensor.S4: (Gas.NO2,),
    Sensor.S5: (),  # no O3 reference
}
# Dashed edges: cross-sensitivities, as drawn in figures 2–3.
# TODO(EDA): confirm with Spearman sensor–gas correlations; keep 1–2 per sensor.
CROSS: dict[Sensor, tuple[Gas, ...]] = {
    Sensor.S2: (Gas.C6H6,),
    Sensor.S4: (Gas.CO,),
    Sensor.S5: (Gas.NO2,),
}


@dataclass(frozen=True)
class Network:
    """The structure of a discrete Bayesian network, as one object.

    Everything a model needs to know about the graph travels together, so a model
    takes one argument (DiscreteBN(network, target)) and each notebook builds its
    phase as a Network, usually through single_gas_network / multi_gas_network:

        network = Network(
            structure={Gas.CO: (HOUR,), **{s: (Gas.CO, *WEATHER) for s in SENSORS}},
            cards={HOUR: 24, Gas.CO: 3, **{w: 3 for w in WEATHER}, **{s: 3 for s in SENSORS}},
            evidence=(HOUR, *WEATHER, *SENSORS),
        ).validate()

    Enum members are strings, so plain names ("CO(GT)", "hour") work as keys too.

    Node kinds
      modeled   a key of `structure`; gets a CPT
      input     appears only as a parent (hour, weather); conditioned on, no CPT
      latent    modeled, but never a data column (Phase 3's Gas.CO.latent == "Gas_CO")
    """

    structure: dict[str, tuple[str, ...]]  # node → parents in the same hour
    cards: dict[str, int]  # number of states for every modeled node and input
    evidence: tuple[str, ...]  # columns visible at predict time; never a reference
    latent: tuple[str, ...] = ()  # modeled nodes with no data column
    prev: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Phase 4: node → parents from the previous hour,
    # e.g. {Gas.CO.latent: (Gas.CO.latent,)}. Empty for a static network.

    @property
    def nodes(self) -> tuple[str, ...]:
        """Modeled nodes in topological order (every parent before its children)."""
        raise NotImplementedError

    @property
    def inputs(self) -> tuple[str, ...]:
        """Names that appear as a parent but are not keys of `structure`."""
        raise NotImplementedError

    @property
    def observed(self) -> tuple[str, ...]:
        """Modeled nodes that are data columns (nodes minus latent)."""
        raise NotImplementedError

    @property
    def is_temporal(self) -> bool:
        """True when any node has parents in the previous hour."""
        raise NotImplementedError

    def children(self, node: str) -> tuple[str, ...]:
        """Modeled nodes that list `node` as a same-hour parent."""
        raise NotImplementedError

    def shape(self, node: str) -> tuple[int, ...]:
        """CPT table shape for `node`: (*parent cards, node card), parents in
        `structure` order. For a temporal node, its previous-hour parents come first."""
        raise NotImplementedError

    def validate(self, columns: Iterable[str] | None = None) -> Self:
        """Raise ValueError listing every problem, else return self (for chaining).

        Checks: every parent is modeled or an input; no cycle within an hour;
        every modeled node and input has a card; latent nodes are not in evidence;
        `prev` only names modeled nodes. With `columns` (a DataFrame's columns),
        also check every observed node and input is a column.
        """
        raise NotImplementedError


def sensor_gases(sensor: Sensor, gases: Sequence[Gas], cross: bool) -> tuple[Gas, ...]:
    """The gas parents of `sensor`: its NOMINAL gas, plus its CROSS gases when
    cross=True, keeping only gases that are in `gases`, nominal first, no repeats.

    e.g. sensor_gases(Sensor.S4, GASES, cross=True)  → (Gas.NO2, Gas.CO)
         sensor_gases(Sensor.S2, GASES, cross=False) → ()   (NMHC not modeled)
    """
    raise NotImplementedError


def single_gas_network(
    target: Gas = Gas.CO,
    weather: Sequence[Weather] = WEATHER,
    n_bins: int = 3,
) -> Network:
    """Phase 1: Hour → target → S1…S5 ← weather.

    All five sensors respond to the one gas; there is no cross-sensitivity yet.

    structure  {target: (HOUR,), every sensor: (target, *weather)}
    cards      HOUR: 24; target, every sensor and every weather column: n_bins
    evidence   (HOUR, *weather, *SENSORS); never the target
    latent     ()
    With weather=(), each sensor's only parent is the target.
    Returns the Network after .validate().
    """
    raise NotImplementedError


def multi_gas_network(
    gases: Sequence[Gas] = GASES,
    weather: Sequence[Weather] = WEATHER,
    cross: bool = True,
    n_bins: int = 3,
) -> Network:
    """Phase 2: Hour → every gas → sensors (nominal + cross) ← weather.

    Every gas is its reference reading, treated as observed truth.

    structure  every gas: (HOUR,); every sensor: (*sensor_gases(s, gases, cross), *weather)
    cards      HOUR: 24; every gas, sensor and weather column: n_bins
    evidence   (HOUR, *weather, *SENSORS); no reference, not even other gases',
               or the score leaks
    latent     ()
    cross=False is the nominal-only ablation. S5 then has no gas parent and depends
    on weather only. S2 has no gas parent at all, since NMHC isn't modeled.
    Returns the Network after .validate().
    """
    raise NotImplementedError
