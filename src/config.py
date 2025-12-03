"""
Application configuration management.
"""

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "GivingIntelligence"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/giving_intelligence"
    redis_url: str = "redis://localhost:6379/0"

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: str = "anthropic"  # "openai", "anthropic", or "mock"

    # Vector Store
    chroma_persist_directory: str = "./chroma_data"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Authentication
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Agent Settings
    agent_timeout_seconds: int = 30
    max_recommendations_per_request: int = 10

    # External APIs
    guidestar_api_key: str = ""
    charity_navigator_api_key: str = ""

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

