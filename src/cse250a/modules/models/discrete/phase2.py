from cse250a.modules.models.discrete.bn import DiscreteBN

GASES = ["CO(GT)", "NMHC(GT)", "NOx(GT)", "NO2(GT)"]

NOMINAL = {  # solid edges: each sensor's target gas
    "PT08.S1(CO)": ["CO(GT)"],
    "PT08.S2(NMHC)": ["NMHC(GT)"],
    "PT08.S3(NOx)": ["NOx(GT)"],
    "PT08.S4(NO2)": ["NO2(GT)"],
    "PT08.S5(O3)": [],  # no O3 reference
}

# TODO: we need to update this.
CROSS = {  # dashed edges: from the EDA joint regressions
    "PT08.S4(NO2)": ["CO(GT)"],
    "PT08.S5(O3)": ["NO2(GT)"],
}


class Phase2(DiscreteBN):
    """All gases: Hour → each gas → sensors (nominal + cross-sensitivity) ← weather."""

    name = "phase2"

    def __init__(
        self,
        target: str = "CO(GT)",
        weather: list[str] | None = None,
        cross: bool = True,
        alpha: float = 1.0,
        hour_col: str = "hour",
    ):
        super().__init__(target, alpha)
        self.weather = weather or ["T", "AH"]
        self.cross = cross  # False = nominal-only ablation
        self.hour_col = hour_col

    @property
    def structure(self) -> dict[str, list[str]]:
        """Phase 2 network: Hour → each gas → sensors (nominal + cross-sensitivity) ← weather.

        Every gas in ``GASES`` depends on hour of day. Each sensor depends on its
        nominal gas (``NOMINAL``), any cross-sensitivity gases (``CROSS``, only
        when ``cross=True``), and the weather variables. S5 has no nominal gas
        (there is no O3 reference), so with ``cross=False`` it depends on weather
        only and carries no information about any gas.

        Returns:
            dict[str, list[str]]: Node name → list of parent names. Keys are every
            gas and every sensor; ``hour`` and the weather columns are root nodes
            and have no entries.
        """
        gases = {g: [self.hour_col] for g in GASES}
        sensors = {
            s: [*NOMINAL[s], *(CROSS.get(s, []) if self.cross else []), *self.weather]
            for s in NOMINAL
        }
        return gases | sensors
