from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTORIST_", extra="ignore")

    cache_path: Path
    vec_path: Path
    vector_backend: str = "sqlite"
    database_url: str | None = None
    embedder: str = "fastembed"
    enricher: str = "none"
    claude_model: str = "claude-haiku-5-5"
    enrich_poll_seconds: float = 2.0
    enrich_max_attempts: int = 3
    max_asset_bytes: int = 5_000_000
