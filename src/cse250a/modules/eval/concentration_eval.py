"""One-call 'Concentration evaluation' section shared by every phase notebook."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cse250a.modules.eval.concentration import (
    bin_means,
    concentration_error,
    expected_concentration,
    resolution_floor,
)
from cse250a.modules.eval.plotting import UNITS, plot_concentration_trace, plot_mae_bars
from cse250a.modules.eval.skill import skill_table

WEEK = slice("2004-05-10", "2004-05-16") #To-do: confirm this is a good week for all phases


def _ms(s):
    return f"{s.mean():.3f} ± {s.std():.3f}"


def concentration_section(*, gases, grid, est_val, val_raw, c_bar, edges, week = WEEK,
                          phase_label, n_bins, fig_path):
    """
    gases   : gas keys (e.g. Gas.CO)
    grid    : long DataFrame with columns bins, seed, gas, model, MAE, RMSE;
              models named BN, HourOnlyBaseline, resolution_floor
    est_val : {gas: Series} posterior-mean estimates on val (one seed)
    val_raw : DataFrame of raw val readings (same seed)
    c_bar   : {gas: training bin means}
    edges   : {gas: interior bin edges}
    week    : slice of dates, identical in every notebook
    fig_path: callable filename -> path
    """
    mae = grid.pivot_table(index="gas", columns="model", values="MAE", aggfunc="mean")
    for g in gases:
        name = str(g).split("(")[0]
        plot_concentration_trace(
            val_raw.loc[week, g], est_val[g].loc[week], np.asarray(c_bar[g]), edges[g],
            gas=name, unit=UNITS[name], phase=phase_label,
            path=fig_path(f"concentration_trace_{name}.png"),
        )
        plt.show()
        plot_mae_bars(
            mae.loc[str(g), "HourOnlyBaseline"], mae.loc[str(g), "BN"],
            mae.loc[str(g), "resolution_floor"],
            gas=name, unit=UNITS[name], phase=phase_label, n_bins=n_bins,
            path=fig_path(f"concentration_mae_{name}.png"),
        )
        plt.show()

    skill = skill_table(grid)
    rmse = grid.pivot_table(index=["gas", "seed"], columns="model", values="RMSE")
    rows = []
    for gas, d in skill.groupby("gas"):
        r = rmse.loc[gas]
        rows.append({
            "gas": gas, "bins": n_bins,
            "BN MAE": _ms(d["BN"]), "hour MAE": _ms(d["HourOnlyBaseline"]),
            "floor MAE": _ms(d["resolution_floor"]),
            "BN RMSE": _ms(r["BN"]), "hour RMSE": _ms(r["HourOnlyBaseline"]),
            "skill_vs_baseline": _ms(d["skill_vs_baseline"]),
            "gap_closed": _ms(d["gap_closed"]),
        })
    return pd.DataFrame(rows).set_index("gas")


def class_metrics(proba, true):
    m = true.notna()
    y = true[m].astype(int).to_numpy()
    P = proba.loc[m].to_numpy()
    pred = P.argmax(axis=1)
    ll = np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))
    return {
        "accuracy": (pred == y).mean(),
        "within1": (np.abs(pred - y) <= 1).mean(),
        "mean_ll": ll.mean(),
    }


def make_bin_means(train_raw, train, gases):
    return {g: bin_means(train_raw[g], train[g]) for g in gases}


def estimate_all(model, binned, c_bar, evidence, gases):
    est, proba = {}, {}
    for g in gases:
        p = model.predict_proba(binned, g, evidence)
        proba[g] = p
        est[g] = expected_concentration(p, c_bar[g])
    return pd.DataFrame(est, index=binned.index), proba


def build_cache(models, splits, which, evidence, gases):
    out = {}
    for seed, sp in splits.items():
        c_bar = make_bin_means(sp["train_raw"], sp["train"], gases)
        out[seed] = {"c_bar": c_bar}
        for name, m in models[seed].items():
            est, proba = estimate_all(m, sp[which], c_bar, evidence, gases)
            out[seed][name] = {"est": est, "proba": proba}
    return out


def per_gas_table(cache, splits, which, gases, model_names=("event", "phase2", "hour")):
    rows = []
    for seed, sp in splits.items():
        raw, binned, c_bar = sp[f"{which}_raw"], sp[which], cache[seed]["c_bar"]
        for g in gases:
            for name in model_names:
                d = cache[seed][name]
                rows.append({
                    "seed": seed, "gas": str(g), "model": name,
                    **class_metrics(d["proba"][g], binned[g]),
                    **concentration_error(d["est"][g], raw[g]),
                })
            rows.append({
                "seed": seed, "gas": str(g), "model": "floor",
                **resolution_floor(raw[g], binned[g], c_bar[g]),
            })
    return pd.DataFrame(rows)