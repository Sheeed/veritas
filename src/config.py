"""Zentrale Konfiguration für History Guardian."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API Key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI Model to use")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="historyguardian2024")

    # Application
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
