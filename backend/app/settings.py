"""Application settings loaded from environment variables."""

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
    CHAT_ENABLED: bool = False
    LLM_BASE_URL: str | None = None
    LLM_API_KEY: str | None = None
    LLM_MODEL: str | None = None
    LLM_PROVIDER_LABEL: str = "LLM"
    CHAT_RATE_LIMIT: str = "10/minute"
    CHAT_DAILY_CAP: int = 500
    LOG_LEVEL: str = "INFO"


settings = Settings()
