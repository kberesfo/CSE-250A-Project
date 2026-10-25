# modules/models/discrete/phase1.py
from .bn import DiscreteBN

SENSORS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
]


class Phase1(DiscreteBN):
    """Single-gas BN: Hour → target → S1…S5 ← weather."""

    name = "phase1"

    def __init__(
        self,
        target: str = "CO(GT)",
        weather: list[str] | None = None,
        alpha: float = 1.0,
        hour_col: str = "hour",
    ):
        super().__init__(target, alpha)
        self.weather = weather or ["T", "AH"]
        self.hour_col = hour_col

    @property
    def structure(self) -> dict[str, list[str]]:
        """Phase 1 network: Hour → target → S1…S5 ← weather.

        The target gas depends only on hour of day. Every sensor depends on the
        target gas and the weather variables, so all five sensors are treated as
        responding to the one gas (no cross-sensitivity until Phase 2). With
        ``weather=[]``, each sensor has the target as its only parent.

        Returns:
            dict[str, list[str]]: Node name → list of parent names. Keys are the
            target and the five sensor columns; ``hour`` and the weather columns
            are root nodes and have no entries.
        """
        return {
            self.target: [self.hour_col],
            **{s: [self.target, *self.weather] for s in SENSORS},
        }
