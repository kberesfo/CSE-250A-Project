"""Unitless comparison metrics for concentration estimates.

Both metrics compare a model's error against the hour-only baseline and the
perfect-bin resolution floor, so results are comparable across gases with
different units (CO in mg/m^3, NOx in ppb, NO2 in ug/m^3) and across phases.
"""

import pandas as pd


def skill_vs_baseline(mae_model: float, mae_baseline: float) -> float:
    """Fraction of baseline MAE removed by the model: 1 - MAE_model / MAE_baseline.

    0 = no better than baseline, 1 = zero error, < 0 = worse than baseline.
    NaN when the baseline error is zero or undefined.
    """
    if not mae_baseline > 0:  # also catches NaN
        return float("nan")
    return 1.0 - mae_model / mae_baseline


def gap_closed(mae_model: float, mae_baseline: float, mae_floor: float) -> float:
    """Fraction of the achievable improvement (baseline -> resolution floor) captured.

    0 = baseline-level, 1 = as good as knowing the true bin. Can slightly exceed 1,
    since a hedged posterior may beat the true bin's mean on some hours.
    NaN when the baseline is already at (or below) the floor, i.e. no achievable gap.
    """
    gap = mae_baseline - mae_floor
    if not gap > 0:  # also catches NaN
        return float("nan")
    return (mae_baseline - mae_model) / gap


def skill_table(
    grid: pd.DataFrame,
    model: str = "BN",
    baseline: str = "HourOnlyBaseline",
    floor: str = "resolution_floor",
    metric: str = "MAE",
) -> pd.DataFrame:
    """Per-(bins, seed, gas) skill and gap-closed from a long-format error grid.

    `grid` needs columns: bins, seed, gas, model, and `metric`, with one row per
    (bins, seed, gas, model), as built in the phase notebooks.
    """
    required = {"bins", "seed", "gas", "model", metric}
    missing = required - set(grid.columns)
    if missing:
        raise KeyError(f"grid is missing columns: {sorted(missing)}")

    wide = grid.pivot_table(
        index=["bins", "seed", "gas"], columns="model", values=metric
    )
    for name in (model, baseline, floor):
        if name not in wide.columns:
            raise KeyError(f"model {name!r} not found in grid['model']")

    wide["skill_vs_baseline"] = [
        skill_vs_baseline(m, b) for m, b in zip(wide[model], wide[baseline])
    ]
    wide["gap_closed"] = [
        gap_closed(m, b, f)
        for m, b, f in zip(wide[model], wide[baseline], wide[floor])
    ]
    wide.columns.name = None
    return wide.reset_index()