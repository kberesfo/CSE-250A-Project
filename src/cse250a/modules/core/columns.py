# modules/core/columns.py
from enum import StrEnum

HOUR = "hour"  # added by load_air_quality


class Gas(StrEnum):
    """Reference-analyzer columns (Ref_g in the ROADMAP)."""

    CO = "CO(GT)"
    C6H6 = "C6H6(GT)"
    NOX = "NOx(GT)"
    NO2 = "NO2(GT)"
    NMHC = "NMHC(GT)"  # in the data, ~90% missing; not modeled by default

    @property
    def latent(self) -> str:
        """Name of the latent true level (Phase 3+): Gas.CO.latent == "Gas_CO"."""
        return "Gas_" + self.value.split("(")[0]


class Sensor(StrEnum):
    """Low-cost sensor columns. The name in parentheses is the nominal target."""

    S1 = "PT08.S1(CO)"
    S2 = "PT08.S2(NMHC)"
    S3 = "PT08.S3(NOx)"
    S4 = "PT08.S4(NO2)"
    S5 = "PT08.S5(O3)"


class Weather(StrEnum):
    T = "T"
    RH = "RH"
    AH = "AH"
