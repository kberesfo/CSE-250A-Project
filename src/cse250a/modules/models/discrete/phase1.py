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
        return {
            self.target: [self.hour_col],
            **{s: [self.target, *self.weather] for s in SENSORS},
        }
