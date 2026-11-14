from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# src/cse250a/modules/config/paths.py -> repo root is 4 levels up
REPO_ROOT = Path(__file__).resolve().parents[4]


from pathlib import Path

# src/cse250a/modules/config/paths.py -> repo root is 4 levels up
REPO_ROOT = Path(__file__).resolve().parents[4]


class PathSettings(BaseSettings):
    """Filesystem locations. Override with DATA_DIR, RESULTS_DIR, FIGURES_DIR."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"
    figures_dir: Path = REPO_ROOT / "figures"

    @property
    def air_quality_csv(self) -> Path:
        return self.data_dir / "air_quality" / "AirQualityUCI.csv"

    def figure(self, notebook: str, name: str) -> Path:
        """Path for a generated figure: figures/<notebook>/<name>. Creates the folder."""
        return self._output(self.figures_dir, notebook, name)

    def result(self, notebook: str, name: str) -> Path:
        """Path for a generated table or file: results/<notebook>/<name>. Creates the folder."""
        return self._output(self.results_dir, notebook, name)

    @staticmethod
    def _output(root: Path, notebook: str, name: str) -> Path:
        # `notebook` is the notebook's folder relative to notebooks/,
        # e.g. "phase1" or "studies/cross_sensitivity".
        if Path(notebook).is_absolute():
            raise ValueError(
                f"notebook must be relative (e.g. 'phase1'), got {notebook!r}"
            )
        d = root / notebook
        d.mkdir(parents=True, exist_ok=True)
        return d / name
