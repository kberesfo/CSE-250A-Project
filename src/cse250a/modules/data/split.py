import numpy as np
import pandas as pd


def blocked_split(
    df, train=0.5, val=0.25, block="7D", seed=0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle contiguous blocks of time, then assign them to train / val / test."""

    block_id = (df.index - df.index[0]) // pd.Timedelta(block)
    blocks = np.random.default_rng(seed).permutation(block_id.unique())
    n_train = round(len(blocks) * train)
    n_val = round(len(blocks) * val)
    part = {b: "train" for b in blocks[:n_train]}
    part |= {b: "val" for b in blocks[n_train : n_train + n_val]}
    part |= {b: "test" for b in blocks[n_train + n_val :]}
    labels = block_id.map(part)

    return df[labels == "train"], df[labels == "val"], df[labels == "test"]
