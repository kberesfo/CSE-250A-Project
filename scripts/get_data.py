"""Ensure project datasets exist in DATA_DIR (/data in the devcontainer).

Runs automatically on container creation. Safe to re-run: datasets already
present are skipped.

    uv run python scripts/get_data.py            # fetch anything missing
    uv run python scripts/get_data.py --list     # show status
"""
import sys
import urllib.request
import zipfile

from cse250a.paths import DATA_DIR

# name -> (url or None, note). None = manual download (login/form required).
DATASETS = {
    "ai4i": (
        "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        "UCI AI4I 2020 Predictive Maintenance",
    ),
    "air_quality": (
        "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        "UCI Air Quality",
    ),
    "cmapss": (None, "NASA C-MAPSS: download manually, unzip into /data/cmapss/"),
    "mlb": (None, "MLB Elo (Kaggle): download manually, unzip into /data/mlb/"),
    "dominicks": (None, "Dominick's (Kilts Center): download manually into /data/dominicks/"),
}


def present(name: str) -> bool:
    d = DATA_DIR / name
    return d.is_dir() and any(d.iterdir())


def fetch(name: str) -> None:
    url, note = DATASETS[name]
    if present(name):
        print(f"[ok]     {name}")
        return
    if url is None:
        print(f"[manual] {name}: {note}")
        return
    dest = DATA_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{name}.zip"
    try:
        print(f"[get]    {name}: {note}")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        print(f"[done]   {name}")
    except Exception as e:  # don't break container setup over a flaky download
        print(f"[warn]   {name}: download failed ({e}); re-run this script later")
    finally:
        zip_path.unlink(missing_ok=True)


def main() -> None:
    if "--list" in sys.argv:
        for name, (url, note) in DATASETS.items():
            status = "present" if present(name) else ("missing" if url else "manual")
            print(f"  {name:12s} {status:8s} {note}")
        return
    for name in DATASETS:
        fetch(name)


if __name__ == "__main__":
    main()