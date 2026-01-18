from dotenv import load_dotenv
load_dotenv()

"""
LLM Client Abstraktion fuer History Guardian.

Unterstuetzt:
- OpenAI (GPT-4o, GPT-4o-mini)
- Groq (Llama 3.3 70B, Mixtral) - KOSTENLOS
- Mistral
"""

import logging
from enum import Enum
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Unterstuetzte LLM Provider."""
    OPENAI = "openai"
    GROQ = "groq"
    MISTRAL = "mistral"


# Provider-spezifische Konfiguration
PROVIDER_CONFIG = {
    LLMProvider.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    },
    LLMProvider.GROQ: {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "models": [
            "llama-3.3-70b-versatile",    # Beste Qualitaet, kostenlos
            "llama-3.1-70b-versatile",    # Sehr gut
            "llama-3.1-8b-instant",       # Schnell, kleiner
            "mixtral-8x7b-32768",         # Gut fuer lange Kontexte
            "gemma2-9b-it",               # Google Gemma
        ],
    },
    LLMProvider.MISTRAL: {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
        "models": ["mistral-large-latest", "mistral-medium", "mistral-small"],
    },
}


class LLMClient:
    """
    Einheitlicher LLM Client fuer verschiedene Provider.
    
    Alle Provider verwenden das OpenAI-kompatible API Format.
    """
    
    def __init__(
        self,
        provider: LLMProvider | str = LLMProvider.GROQ,
        api_key: str | None = None,
        model: str | None = None,
    ):
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        
        self.provider = provider
        self.config = PROVIDER_CONFIG[provider]
        self.model = model or self.config["default_model"]
        
        # API Key aus Umgebung oder Parameter
        if api_key is None:
            api_key = self._get_api_key_from_env()
        
        if not api_key:
            raise ValueError(f"API Key fuer {provider.value} nicht gefunden")
        
        # OpenAI-kompatibler Client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.config["base_url"],
        )
        
        logger.info(f"LLM Client initialisiert: {provider.value} / {self.model}")
    
    def _get_api_key_from_env(self) -> str | None:
        """Holt API Key aus Umgebungsvariablen."""
        import os
        
        key_names = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.MISTRAL: "MISTRAL_API_KEY",
        }
        
        return os.getenv(key_names.get(self.provider, ""))
    
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: dict | None = None,
    ) -> str:
        """
        Fuehrt eine Chat Completion aus.
        
        Args:
            messages: Liste von {"role": "...", "content": "..."}
            temperature: Kreativitaet (0.0 = deterministisch)
            max_tokens: Maximale Antwortlaenge
            response_format: Optional {"type": "json_object"}
        
        Returns:
            Antwort-Text
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # JSON Mode - nicht alle Provider unterstuetzen das
        if response_format and self.provider in [LLMProvider.OPENAI, LLMProvider.GROQ]:
            kwargs["response_format"] = response_format
        
        response = await self.client.chat.completions.create(**kwargs)
        
        return response.choices[0].message.content or ""
    
    async def extract_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Extrahiert strukturiertes JSON aus der Antwort.
        """
        import json
        import re
        
        # Versuche JSON Mode
        try:
            content = await self.chat_completion(
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Fallback ohne JSON Mode
            content = await self.chat_completion(
                messages=messages,
                temperature=temperature,
            )
        
        # Parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Versuche JSON aus Text zu extrahieren
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Konnte JSON nicht parsen: {content[:200]}")
    
    def available_models(self) -> list[str]:
        """Gibt verfuegbare Modelle fuer den Provider zurueck."""
        return self.config["models"]


def get_llm_client(
    provider: str | None = None,
    model: str | None = None,
) -> LLMClient:
    """
    Factory-Funktion fuer LLM Client.
    
    Liest Provider aus Umgebung wenn nicht angegeben.
    """
    import os
    
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "groq")
    
    return LLMClient(provider=provider, model=model)
