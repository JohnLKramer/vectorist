from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTORIST_", extra="ignore")

    cache_path: Path
    vec_path: Path
    vector_backend: str = "sqlite"
    database_url: str | None = None
    embedder: str = "fastembed"
