# modules/eval/metrics.py
import numpy as np
import pandas as pd


def score(y_true: pd.Series, proba: pd.DataFrame) -> dict:
    """
    Metrics for binned predictions. `proba` columns are bin labels; rows align with y_true.

    Args:
        y_true (pd.Series): _description_
        proba (pd.DataFrame): _description_

    Returns:
        dict: _description_
    """

    y = y_true.astype(int).to_numpy()
    p = proba.to_numpy()
    cols = proba.columns.to_numpy()
    col_idx = np.searchsorted(cols, y)  # map each true bin to its column
    p_true = p[np.arange(len(y)), col_idx]
    y_pred = cols[p.argmax(axis=1)]

    labels = cols
    f1 = []
    for k in labels:
        tp = np.sum((y_pred == k) & (y == k))
        fp = np.sum((y_pred == k) & (y != k))
        fn = np.sum((y_pred != k) & (y == k))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))

    return {
        "n": len(y),
        "accuracy": float(np.mean(y_pred == y)),
        "macro_f1": float(np.mean(f1)),
        "log_likelihood": float(np.log(p_true).sum()),
        "mean_log_likelihood": float(
            np.log(p_true).mean()
        ),  # comparable across splits of different size
        "confusion": pd.crosstab(
            pd.Series(y, name="true"), pd.Series(y_pred, name="pred")
        ).to_dict(),
    }
