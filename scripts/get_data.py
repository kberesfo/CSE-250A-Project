"""Ensure project datasets exist in config.paths.data_dir (<repo>/data by default).

Safe to re-run: datasets already present are skipped.

    uv run python scripts/get_data.py                # fetch anything missing
    uv run python scripts/get_data.py air_quality    # fetch one dataset
    uv run python scripts/get_data.py --list         # show status
    uv run python scripts/get_data.py --force        # re-download, even if present
"""

import shutil
import sys
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cse250a.modules.config import config

DATA_DIR = config.paths.data_dir


@dataclass(frozen=True)
class Dataset:
    note: str
    url: str | None = None  # None = manual download (login/form/agreement required)
    citation: str = ""
    check: Callable[[Path], str] | None = None  # quick parse; returns a one-line summary


# ---------------------------------------------------------------------------
# Sanity checks: confirm the files parse. Not the real loaders (those live in src/).
# ---------------------------------------------------------------------------


def check_air_quality(dest: Path) -> str:
    # File quirks: ';' separator, ',' decimal, trailing empty columns/rows, -200 = missing.
    df = pd.read_csv(dest / "AirQualityUCI.csv", sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    n_missing = int((df == -200).sum().sum())
    return f"{len(df):,} rows x {df.shape[1]} cols, {n_missing:,} cells coded -200 (missing)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, Dataset] = {
    "air_quality": Dataset(
        note="UCI Air Quality (project dataset)",
        url="https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        citation="Vito, S. (2008). Air Quality [Dataset]. UCI ML Repository. "
        "https://doi.org/10.24432/C59K5F (CC BY 4.0)",
        check=check_air_quality,
    ),
    # Backup project (round-2 runner-up). Public download, see E-TRIALS data page.
    # "assist_2009": Dataset(
    #     note="ASSISTments 2009-10 (backup): download manually into data/assist_2009/ "
    #     "from https://www.etrialstestbed.org/data-sets/publicly-available-data-sets",
    # ),
}


# ---------------------------------------------------------------------------


def present(name: str) -> bool:
    d = DATA_DIR / name
    return d.is_dir() and any(d.iterdir())


def fetch(name: str, force: bool = False) -> None:
    ds = DATASETS[name]
    dest = DATA_DIR / name

    if present(name) and not force:
        print(f"[ok]     {name}")
    elif ds.url is None:
        print(f"[manual] {name}: {ds.note}")
        return
    else:
        if force and dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        zip_path = dest / f"{name}.zip"
        try:
            print(f"[get]    {name}: {ds.note}")
            urllib.request.urlretrieve(ds.url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
            print(f"[done]   {name}")
        except (OSError, zipfile.BadZipFile) as e:
            # one flaky download shouldn't stop the others
            print(f"[warn]   {name}: download failed ({e}); re-run this script later")
            return
        finally:
            zip_path.unlink(missing_ok=True)

    if ds.check is not None:
        try:
            print(f"[info]   {name}: {ds.check(dest)}")
        except (OSError, ValueError, KeyError) as e:  # missing file / parse error / bad column
            print(f"[warn]   {name}: sanity check failed ({e})")


def main(argv: list[str]) -> None:
    flags = {a for a in argv if a.startswith("--")}
    names = [a for a in argv if not a.startswith("--")] or list(DATASETS)

    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"Unknown dataset(s): {', '.join(unknown)}. Options: {', '.join(DATASETS)}")

    if "--list" in flags:
        print(f"DATA_DIR = {DATA_DIR}")
        for name in names:
            ds = DATASETS[name]
            status = "present" if present(name) else ("missing" if ds.url else "manual")
            print(f"  {name:12s} {status:8s} {ds.note}")
            if ds.citation:
                print(f"  {'':12s} {'':8s} cite: {ds.citation}")
        return

    for name in names:
        fetch(name, force="--force" in flags)


if __name__ == "__main__":
    main(sys.argv[1:])