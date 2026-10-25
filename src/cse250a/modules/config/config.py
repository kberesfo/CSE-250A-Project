from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .experiment import BinSettings, SplitSettings
from .paths import REPO_ROOT, PathSettings


# config.py
class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env", extra="ignore", env_nested_delimiter="__"
    )  # SPLIT__TRAIN=0.6, BINS__N_BINS=5
    seed: int = 0
    paths: PathSettings = Field(default_factory=PathSettings)
    split: SplitSettings = Field(default_factory=SplitSettings)
    bins: BinSettings = Field(default_factory=BinSettings)

config = Config()
