# modules/eval/metrics.py
import numpy as np
import pandas as pd


def calc_f1(predicted: np.ndarray, y: np.ndarray, k: int) -> float:
    """
    F1 score for one bin, treating bin `k` as positive and every other bin
    as negative (one-vs-rest).

        F1_k = 2·TP / (2·TP + FP + FN)

    where TP = predicted k and truly k, FP = predicted k but truly another bin,
    and FN = truly k but predicted another bin. This is the harmonic mean of
    precision TP / (TP + FP) and recall TP / (TP + FN); true negatives don't
    enter. Averaging over every bin gives macro F1.

    Args:
        predicted (np.ndarray): Predicted bin per row, shape (n,).
        y (np.ndarray): True bin per row, shape (n,), aligned with `predicted`.
        k (int): The bin to score.

    Returns:
        float: F1 in [0, 1]. 0.0 when TP = 0, including when bin k is neither
            predicted nor present.
    """

    TP = np.sum((predicted == k) & (y == k))
    FP = np.sum((predicted == k) & (y != k))
    FN = np.sum((predicted != k) & (y == k))

    return 0.0 if TP == 0 else 2 * TP / (2 * TP + FP + FN)


def calc_BER(predicted: np.ndarray, y: np.ndarray, k: int) -> float:
    """_summary_

    Args:
        predicted (np.ndarray): Predicted bin per row, shape (n,).
        y (np.ndarray): True bin per row, shape (n,), aligned with `predicted`.
        k (int): The bin to score.

    Returns:
        float: BERE in [0,1]
    """

    TP = np.sum((predicted == k) & (y == k))
    FP = np.sum((predicted == k) & (y != k))
    FN = np.sum((predicted != k) & (y == k))
    TN = np.sum((predicted != k) & (y != k))

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    return 1 - 1 / 2 * (TPR + TNR)


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
    bere = []
    for k in labels:
        f1.append(calc_f1(y_pred, y, k))
        bere.append(calc_BER(y_pred, y, k))

    return {
        "n": len(y),
        "accuracy": float(np.mean(y_pred == y)),
        "macro_f1": float(np.mean(f1)),
        "bere": float(np.mean(bere)),
        "log_likelihood": float(np.log(p_true).sum()),
        "mean_log_likelihood": float(
            np.log(p_true).mean()
        ),  # comparable across splits of different size
        "confusion": pd.crosstab(
            pd.Series(y, name="true"), pd.Series(y_pred, name="pred")
        ).to_dict(),
    }
