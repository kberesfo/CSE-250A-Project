"""The toy network shared by the bn_tour notebooks.

Every variable is binary (0 = low, 1 = high) so every number can be checked by
hand. The CPTs are picked by hand, not learned, so the notebooks can test each
DiscreteBN method against numbers we already know.

    hour ──► CO ──► S1 ◄── T
      │       ╲
      └─► NO2 ─►S4          (S4 is cross-sensitive to CO: the loop that makes
                              CO and NO2 compete to explain a high S4)
"""

import numpy as np
import pandas as pd

from cse250a.modules.core import CPT, Gas, Network, Sensor, Time, Weather
from cse250a.modules.models import DiscreteBN

H, T = Time.HOUR, Weather.T  # inputs: always observed, no CPT
CO, NO2 = Gas.CO, Gas.NO2  # references: often missing, never evidence
S1, S4 = Sensor.S1, Sensor.S4  # cheap sensors

STRUCTURE = {
    CO: (H,),
    NO2: (H,),
    S1: (CO, T),
    S4: (NO2, CO),
}
CARDS = {H: 2, T: 2, CO: 2, NO2: 2, S1: 2, S4: 2}
EVIDENCE = (H, T, S1, S4)

# Hand-picked CPTs. Axes are (*parents, node), so TABLES[S1][co, t] is the
# distribution of S1 given CO = co and T = t.
TABLES = {
    CO: np.array([[0.8, 0.2],   # hour 0 (night): CO mostly low
                  [0.3, 0.7]]),  # hour 1 (day):   CO mostly high
    NO2: np.array([[0.7, 0.3],
                   [0.4, 0.6]]),
    S1: np.array([[[0.9, 0.1], [0.7, 0.3]],    # CO low,  T = 0 / 1
                  [[0.2, 0.8], [0.1, 0.9]]]),  # CO high, T = 0 / 1
    S4: np.array([[[0.9, 0.1], [0.4, 0.6]],    # NO2 low,  CO low / high
                  [[0.3, 0.7], [0.1, 0.9]]]),  # NO2 high, CO low / high
}

nan = np.nan
# Eight hand-made rows covering every case the methods have to handle.
DATA = pd.DataFrame(
    [
        # H  T   CO   NO2   S1   S4
        [1, 0, 1, 1, 1, 1],          # 0  complete
        [0, 1, 0, 0, 0, 0],          # 1  complete
        [1, 1, nan, 0, 1, 1],        # 2  CO missing
        [0, 0, 1, nan, 1, 0],        # 3  NO2 missing
        [1, 0, nan, nan, 1, 1],      # 4  both references missing (= a prediction)
        [0, 0, 0, 1, nan, 1],        # 5  a sensor missing
        [1, 1, 1, 0, 1, nan],        # 6  the other sensor missing
        [0, 1, nan, nan, nan, 0],    # 7  three missing
    ],
    columns=[H, T, CO, NO2, S1, S4],
    dtype=float,
)


def network() -> Network:
    return Network(STRUCTURE, CARDS).validate(DATA.columns)


def cpts() -> dict[str, CPT]:
    return {n: CPT(n, STRUCTURE[n], TABLES[n]) for n in STRUCTURE}


def toy_bn() -> DiscreteBN:
    """A DiscreteBN with the hand-picked CPTs, without fitting."""
    bn = DiscreteBN(network())
    bn.cpts = cpts()
    return bn


def sample(n: int, seed: int = 0, missing: float = 0.0) -> pd.DataFrame:
    """n rows drawn from the toy CPTs (inputs uniform). Each reference and
    sensor cell is then blanked with probability `missing`."""
    rng = np.random.default_rng(seed)
    draw = lambda p: (p.cumsum(-1) > rng.random(p.shape[:-1] + (1,))).argmax(-1)
    h, t = rng.integers(0, 2, n), rng.integers(0, 2, n)
    co = draw(TABLES[CO][h])
    no2 = draw(TABLES[NO2][h])
    s1 = draw(TABLES[S1][co, t])
    s4 = draw(TABLES[S4][no2, co])
    df = pd.DataFrame({H: h, T: t, CO: co, NO2: no2, S1: s1, S4: s4}, dtype=float)
    modeled = [CO, NO2, S1, S4]
    df[modeled] = df[modeled].mask(rng.random((n, 4)) < missing)
    return df


def same(name: str, mine, theirs, atol: float = 1e-12) -> None:
    """Print ✅ if two arrays (or dicts of arrays) agree, else what differs."""
    if isinstance(mine, dict):
        bad = [k for k in mine if not np.allclose(mine[k], theirs[k], atol=atol)]
        ok = not bad and set(mine) == set(theirs)
        print(f"{'✅' if ok else '❌'} {name}" + ("" if ok else f"  (differs: {bad})"))
        return
    mine, theirs = np.asarray(mine, dtype=float), np.asarray(theirs, dtype=float)
    ok = mine.shape == theirs.shape and np.allclose(mine, theirs, atol=atol)
    detail = "" if ok else f"\n   mine:   {mine.round(6).tolist()}\n   theirs: {theirs.round(6).tolist()}"
    print(f"{'✅' if ok else '❌'} {name}{detail}")
