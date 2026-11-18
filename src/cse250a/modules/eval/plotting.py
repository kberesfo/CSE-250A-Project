"""Shared plotting helpers for the per-phase concentration evaluation section."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"reference": "black", "estimate": "#2E7D32", "baseline": "#9E9E9E", "floor": "#455A64"}


def _band_colors(n_bins):
    """Green -> yellow -> red band shades for any bin count."""
    return plt.get_cmap("RdYlGn_r")(np.linspace(0.15, 0.85, n_bins), alpha=0.18)


def plot_concentration_trace(reference, estimate, bin_means, edges, *,
                             gas, unit, phase, ax=None, path=None):
    """Reference vs. posterior-mean estimate over a window, with bin bands and bin means.

    reference, estimate : Series over the same time window
    bin_means           : training bin means (one per bin, ascending)
    edges               : interior bin edges (len = n_bins - 1)
    """
    n_bins = len(bin_means)
    if len(edges) != n_bins - 1:
        raise ValueError(f"expected {n_bins - 1} interior edges, got {len(edges)}")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    ax.plot(reference, color=COLORS["reference"], lw=1.2, label="reference (ground truth)")
    ax.plot(estimate, color=COLORS["estimate"], lw=1.2, label="posterior-mean estimate")
    for c in bin_means:
        ax.axhline(c, color="grey", ls=":", lw=0.8)

    y0, y1 = ax.get_ylim()
    bounds = [y0, *np.clip(edges, y0, y1), y1]
    for i, color in enumerate(_band_colors(n_bins)):
        ax.axhspan(bounds[i], bounds[i + 1], color=color, zorder=0)
    ax.set_ylim(y0, y1)

    ax.set_ylabel(f"{gas} ({unit})")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{phase}: {gas}, {n_bins} bins (estimate is limited to the dotted bin means)")
    if path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(path, bbox_inches="tight")
    return ax


def plot_mae_bars(mae_baseline, mae_model, mae_floor, *,
                  gas, unit, phase, n_bins, ax=None, path=None):
    """Horizontal MAE bars: hour-only baseline, phase model, resolution floor."""
    mae = pd.Series({
        "hour-only baseline": mae_baseline,
        f"{phase} BN": mae_model,
        f"{n_bins}-bin floor": mae_floor,
    })
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 2.2))
    mae.plot.barh(ax=ax, color=[COLORS["baseline"], COLORS["estimate"], COLORS["floor"]])
    pad = 0.01 * mae.max()                     # scales with the gas's units
    for i, v in enumerate(mae):
        ax.text(v + pad, i, f"{v:.3g}", va="center", fontsize=9)
    ax.set_xlim(0, mae.max() * 1.15)
    ax.set_xlabel(f"MAE ({unit})")
    ax.set_title(f"{gas} concentration error (validation)")
    if path is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(path, bbox_inches="tight")
    return ax


# Units per the UCI Air Quality dataset documentation
UNITS = {"CO": "mg/m$^3$", "NOx": "ppb", "NO2": r"$\mu$g/m$^3$"}