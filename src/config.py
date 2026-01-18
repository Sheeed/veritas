"""
Konfiguration fuer History Guardian.

Laedt Einstellungen aus Umgebungsvariablen und .env Datei.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Anwendungseinstellungen."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App Info
    app_name: str = "History Guardian"
    app_version: str = "0.3.0"

    # LLM Provider: "openai", "groq", "mistral"
    llm_provider: str = "groq"
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    
    # Groq (KOSTENLOS)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    
    # Mistral
    mistral_api_key: str = ""
    mistral_model: str = "mistral-large-latest"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "historyguardian2024"
    neo4j_database: str = "neo4j"

    # LLM Parameters
    llm_temperature: float = 0.0
    max_tokens: int = 4096

    # Processing
    max_content_length: int = 10000
    extraction_timeout: int = 60

    # Logging
    log_level: str = "INFO"
    debug: bool = False
    
    def get_active_api_key(self) -> str:
        """Gibt den API Key fuer den aktiven Provider zurueck."""
        keys = {
            "openai": self.openai_api_key,
            "groq": self.groq_api_key,
            "mistral": self.mistral_api_key,
        }
        return keys.get(self.llm_provider, "")
    
    def get_active_model(self) -> str:
        """Gibt das Modell fuer den aktiven Provider zurueck."""
        models = {
            "openai": self.openai_model,
            "groq": self.groq_model,
            "mistral": self.mistral_model,
        }
        return models.get(self.llm_provider, "")


@lru_cache
def get_settings() -> Settings:
    """Gibt gecachte Settings-Instanz zurueck."""
    return Settings()
