from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/cse250a/modules/config/paths.py -> repo root is 4 levels up
REPO_ROOT = Path(__file__).resolve().parents[4]


class PathSettings(BaseSettings):
    """Filesystem locations. Override with DATA_DIR, RESULTS_DIR, FIGURES_DIR."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"
    figures_dir: Path = REPO_ROOT / "figures"
