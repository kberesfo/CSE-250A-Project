# modules/eval/concentration.py
"""Turn bin probabilities back into a concentration, and score it in real units.

The BN predicts bins. Each bin stands for a typical concentration (the mean of the
training hours in it), so the natural point estimate is the probability-weighted
average of those values:

    estimate_t = Σ_k P(bin_k | evidence_t) · c̄_k

The estimate is continuous, but its information is still limited by the bins: it
never exceeds the top bin's mean, and even a perfect bin prediction can't beat
`resolution_floor`.

Example:
    from cse250a.modules.eval.concentration import (
        bin_means, concentration_error, expected_concentration, resolution_floor,
    )

    gases = (Gas.CO, Gas.C6H6, Gas.NOX, Gas.NO2)
    evidence = (Time.HOUR, *weather, *Sensor)

    estimates = {}   # gas → estimated concentration per hour
    rows = []
    for gas in gases:
        c_bar = bin_means(train_raw[gas], train[gas])          # this gas's bin means, from train
        for model in (bn, hour):
            proba = model.predict_proba(val, gas, evidence)    # the slow step: once per (model, gas)
            est = expected_concentration(proba, c_bar)
            if model is bn:
                estimates[gas] = est
            rows.append({"gas": str(gas), "model": model.name,
                        **concentration_error(est, val_raw[gas])})
        rows.append({"gas": str(gas), "model": "resolution floor",
                    **resolution_floor(val_raw[gas], val[gas], c_bar)})

    results = pd.DataFrame(rows).set_index(["gas", "model"])
    estimates = pd.DataFrame(estimates)    # one column per gas, one row per hour, in real units
    results.round(3)
"""

import numpy as np
import pandas as pd


def bin_means(raw: pd.Series, binned: pd.Series, stat: str = "mean") -> pd.Series:
    """Typical concentration of each bin: c̄_k, from training data only.

    Args:
        raw: Raw concentrations (e.g. train_raw[Gas.CO]).
        binned: The same rows after Discretizer.transform (bin per row, NaN if missing).
            Must share raw's index.
        stat: "mean" (minimizes RMSE) or "median" (minimizes MAE; better for
            right-skewed gases with rare spikes).

    Returns:
        pd.Series indexed by bin (0 … k-1), sorted. Rows missing in either are ignored.
    """
    if not raw.index.equals(binned.index):
        raise ValueError("raw and binned must share an index (transform the same rows)")
    return raw.groupby(binned).agg(stat).sort_index()


def expected_concentration(proba: pd.DataFrame, c_bar: pd.Series) -> pd.Series:
    """Σ_k P(bin_k) · c̄_k for every row of `proba` (as returned by predict_proba).

    Raises:
        ValueError: If a bin column of `proba` has no entry in `c_bar` (a bin with
            no training hours).
    """
    missing = [k for k in proba.columns if k not in c_bar.index]
    if missing:
        raise ValueError(f"no bin mean for bins {missing}; were they empty in train?")
    values = proba.to_numpy() @ c_bar.loc[proba.columns].to_numpy()
    return pd.Series(values, index=proba.index, name="estimate")


def concentration_error(estimate: pd.Series, truth: pd.Series) -> dict:
    """MAE and RMSE of `estimate` against raw `truth`, on rows where truth is observed.

    Returns:
        {"n", "MAE", "RMSE"}, in the units of `truth`.
    """
    seen = truth.notna() & estimate.notna()
    err = estimate[seen].to_numpy() - truth[seen].to_numpy()
    return {
        "n": int(seen.sum()),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
    }


def resolution_floor(raw: pd.Series, binned: pd.Series, c_bar: pd.Series) -> dict:
    """Error of a *perfect* bin prediction: every hour gets its true bin's c̄_k.

    No model using these bins can beat this. Compare a model's error to it: close
    to the floor means the bins are the limit (try more); far above means the
    model is.
    """
    return concentration_error(binned.map(c_bar), raw)
