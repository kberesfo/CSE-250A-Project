from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import REPO_ROOT, PathSettings


class Config(BaseSettings):
    """Top-level project config. Add new groups as fields (e.g. em: EMSettings)."""

    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    seed: int = 0  # SEED
    paths: PathSettings = Field(default_factory=PathSettings)


config = Config()
