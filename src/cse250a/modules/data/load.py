import numpy as np
import pandas as pd


def default_load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",")
    # two empty trailing cols
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    # blank trailing rows
    df = df.dropna(how="all")
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S"
    )

    df = df.drop(columns=["Date", "Time"]).set_index("datetime")
    df = df.replace(-200, np.nan)

    return df
