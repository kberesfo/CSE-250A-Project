import numpy as np
import pandas as pd
import pytest

from cse250a.modules.eval.skill import gap_closed, skill_table, skill_vs_baseline


def test_skill_vs_baseline():
    assert skill_vs_baseline(0.5, 1.0) == 0.5
    assert skill_vs_baseline(1.0, 1.0) == 0.0          # no better than baseline
    assert skill_vs_baseline(1.5, 1.0) == -0.5         # worse than baseline
    assert np.isnan(skill_vs_baseline(0.5, 0.0))       # undefined baseline
    assert np.isnan(skill_vs_baseline(0.5, np.nan))


def test_gap_closed():
    assert gap_closed(1.0, 1.0, 0.5) == 0.0            # baseline-level
    assert gap_closed(0.5, 1.0, 0.5) == 1.0            # at the floor
    assert gap_closed(0.75, 1.0, 0.5) == 0.5           # halfway
    assert gap_closed(0.4, 1.0, 0.5) > 1.0             # beats the floor (allowed)
    assert np.isnan(gap_closed(0.4, 0.5, 0.5))         # no achievable gap
    assert np.isnan(gap_closed(0.4, 0.5, np.nan))


def _grid():
    rows = []
    for gas, (bn, hour, floor) in {"CO": (0.488, 0.845, 0.40), "NOx": (76.7, 136.0, 60.0)}.items():
        for model, mae in (("BN", bn), ("HourOnlyBaseline", hour), ("resolution_floor", floor)):
            rows.append({"bins": 5, "seed": 0, "gas": gas, "model": model, "MAE": mae})
    return pd.DataFrame(rows)


def test_skill_table_matches_scalar_functions():
    table = skill_table(_grid()).set_index("gas")
    assert table.loc["CO", "skill_vs_baseline"] == pytest.approx(1 - 0.488 / 0.845)
    assert table.loc["NOx", "gap_closed"] == pytest.approx((136.0 - 76.7) / (136.0 - 60.0))
    assert {"bins", "seed", "gas", "skill_vs_baseline", "gap_closed"} <= set(table.reset_index().columns)


def test_skill_table_missing_model_raises():
    grid = _grid()
    with pytest.raises(KeyError):
        skill_table(grid[grid["model"] != "resolution_floor"])