from pathlib import Path

import numpy as np
import pandas as pd


def load_air_quality(path: str | Path) -> pd.DataFrame:
    """UCI Air Quality as an hourly DataFrame: datetime index, -200 → NaN, plus an `hour` column."""
    
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # two empty trailing columns
    df = df.dropna(how="all")  # blank trailing rows
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime").sort_index()
    df = df.replace(-200, np.nan)

    if not df.index.is_unique:
        raise ValueError("duplicate timestamps in the air quality data")

    df["hour"] = df.index.hour  # 0–23, used by Hour → gas

    return df
