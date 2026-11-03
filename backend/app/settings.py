"""Application settings loaded from environment variables."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for OceanScope backend API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    DATA_DIR: str = "/data/outputs"
    HF_DATASET_REPO: str | None = None
    HF_DATASET_REVISION: str = "main"
    HF_TOKEN: str | None = None
    CORS_ORIGINS: str = "http://localhost:3000"
    # Meta features (PRD_BACKEND §3, §4.12, §6)
    META_API_BASE_URL: str | None = None
    META_API_KEY: str | None = None
    META_CHAT_MODEL: str | None = None
    META_CHAT_FALLBACK_MODEL: str | None = None
    META_TRANSCRIBE_MODEL: str | None = None
    META_IMAGE_MODEL: str | None = None
    META_SEGMENT_MODEL: str | None = None
    # "http" calls the Meta Model API; "mock" uses the offline guide (no key, no network).
    META_CLIENT: Literal["http", "mock"] = "http"
    PROVIDER_LABEL: str = "Muse Spark via Meta Model API"
    CHAT_ENABLED: bool = False
    VOICE_ENABLED: bool = False
    POSTCARD_ENABLED: bool = False
    SEGMENT_ENABLED: bool = False
    BEACH_ENABLED: bool = False
    SEGMENT_PROVIDER: Literal["sam", "threshold"] = "sam"
    SEGMENT_THRESHOLD_C: float = 0.5
    CHAT_RATE_LIMIT: str = "10/minute"
    CHAT_DAILY_CAP: int = 500
    VOICE_RATE_LIMIT: str = "6/minute"
    VOICE_DAILY_CAP: int = 300
    POSTCARD_RATE_LIMIT: str = "3/minute"
    POSTCARD_DAILY_CAP: int = 100
    SEGMENT_RATE_LIMIT: str = "10/minute"
    SEGMENT_DAILY_CAP: int = 300
    # Where generated postcards are cached (§4.15). Empty means <system temp>/oceanscope-postcards.
    # Ephemeral on Render is fine; never inside the repository.
    POSTCARD_CACHE_DIR: str = ""
    RELOAD_INTERVAL_MIN: int = 60
    LOG_LEVEL: str = "INFO"


settings = Settings()
