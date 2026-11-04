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
    # Muse Spark always reasons; lower effort means faster, cheaper turns. Empty = model default.
    META_CHAT_REASONING_EFFORT: Literal["", "minimal", "low", "medium", "high", "xhigh"] = "low"
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
    # Request limits per feature: per IP (sliding window), per IP per UTC day, and global per
    # UTC day. Kept tight because the Meta credits are a fixed, small pot.
    CHAT_RATE_LIMIT: str = "4/minute"
    CHAT_IP_DAILY_CAP: int = 30
    CHAT_DAILY_CAP: int = 150
    VOICE_RATE_LIMIT: str = "3/minute"
    VOICE_IP_DAILY_CAP: int = 15
    VOICE_DAILY_CAP: int = 60
    POSTCARD_RATE_LIMIT: str = "1/minute"
    POSTCARD_IP_DAILY_CAP: int = 3
    POSTCARD_DAILY_CAP: int = 20
    SEGMENT_RATE_LIMIT: str = "3/minute"
    SEGMENT_IP_DAILY_CAP: int = 20
    SEGMENT_DAILY_CAP: int = 60
    # Credit budget for real Meta calls (app/meta/budget.py). Fail closed: with META_CLIENT=http
    # nothing is called until the budget, the prices and a durable ledger are all set.
    META_BUDGET_USD: float = 0.0
    META_DAILY_BUDGET_USD: float = 0.0
    META_BUDGET_LEDGER_PATH: str | None = None
    META_BUDGET_HF_REPO: str | None = None
    HF_BUDGET_TOKEN: str | None = None
    # Prices from the Meta dashboard, in USD. None means "unknown", which keeps that call off.
    META_PRICE_INPUT_PER_MTOK: float | None = None
    META_PRICE_OUTPUT_PER_MTOK: float | None = None
    META_PRICE_PER_AUDIO_MIN: float | None = None
    META_PRICE_PER_IMAGE: float | None = None
    META_PRICE_PER_SEGMENT: float | None = None
    # Where generated postcards are cached (§4.15). Empty means <system temp>/oceanscope-postcards.
    # Ephemeral on Render is fine; never inside the repository.
    POSTCARD_CACHE_DIR: str = ""
    RELOAD_INTERVAL_MIN: int = 60
    LOG_LEVEL: str = "INFO"


settings = Settings()
