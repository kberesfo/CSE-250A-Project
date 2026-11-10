# modules/eval/correlation.py
"""Rank correlations for the cross-sensitivity study (Phase 2 edges).

Spearman is used throughout because it depends only on ranks, like quantile
binning, so log-transforming a gas or sensor does not change it.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def spearman_matrix(
    df: pd.DataFrame, rows: Sequence[str], cols: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Spearman ρ between every `rows` column and every `cols` column.

    Each pair uses the hours where both columns are observed, so pairs involving
    a sparse column (NMHC) rest on fewer hours.

    Returns:
        (rho, n): two DataFrames indexed by `rows` with `cols` as columns,
        holding ρ and the number of hours behind it.
    """
    rho = np.full((len(rows), len(cols)), np.nan)
    n = np.zeros((len(rows), len(cols)), dtype=int)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            pair = df[list(dict.fromkeys([r, c]))].dropna()
            n[i, j] = len(pair)
            if r == c:
                rho[i, j] = 1.0
            elif len(pair) > 2:
                rho[i, j] = pair[r].rank().corr(pair[c].rank())
    index, columns = list(rows), list(cols)
    return (
        pd.DataFrame(rho, index=index, columns=columns),
        pd.DataFrame(n, index=index, columns=columns),
    )


def partial_spearman(
    df: pd.DataFrame,
    x: str,
    y: str,
    controls: Sequence[str] = (),
    *,
    categorical: Sequence[str] = (),
) -> tuple[float, int]:
    """Spearman correlation of `x` and `y` after removing what `controls` explain.

    Rank-transform x, y and the numeric controls, regress x and y on the controls
    by least squares, and correlate the residuals. Categorical controls (e.g.
    hour) enter as one-hot dummies, so their effect is not forced to be monotone.

    Uses only rows where x, y and every control are observed.

    Args:
        df: Raw (unbinned) data.
        x, y: The two columns to correlate.
        controls: Columns to hold fixed, e.g. (nominal gas, "T", "AH", "hour").
        categorical: The subset of `controls` to one-hot encode.

    Returns:
        (rho, n): the partial correlation and the number of rows used.
        With no controls this is plain Spearman ρ.
    """
    cols = list(dict.fromkeys([x, y, *controls]))
    d = df[cols].dropna()
    n = len(d)
    if n <= len(controls) + 2:
        return float("nan"), n

    numeric = [c for c in controls if c not in categorical]
    design = [np.ones((n, 1))]
    if numeric:
        design.append(d[numeric].rank().to_numpy(dtype=float))
    for c in categorical:
        design.append(
            pd.get_dummies(d[c].astype(int), drop_first=True).to_numpy(dtype=float)
        )
    a = np.hstack(design)

    def residual(col: str) -> np.ndarray:
        v = d[col].rank().to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(a, v, rcond=None)
        return v - a @ beta

    rx, ry = residual(x), residual(y)
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return (float(rx @ ry / denom) if denom > 0 else float("nan")), n
