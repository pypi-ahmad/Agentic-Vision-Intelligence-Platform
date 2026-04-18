"""Provider registry — factory for building the right LLMProvider."""

from __future__ import annotations

from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from src.providers.gemini_provider import GeminiProvider
from src.providers.ollama_provider import OllamaProvider
from src.providers.openai_provider import OpenAIProvider

__all__ = [
    "get_provider",
    "LLMProvider",
    "ProviderError",
    "ProviderAuthError",
    "ProviderConnectionError",
    "ProviderRateLimitError",
]


def get_provider(name: str, *, api_key: str = "", ollama_url: str = "http://localhost:11434") -> LLMProvider:
    """Instantiate a provider adapter by name.

    Args:
        name: One of 'Ollama', 'OpenAI', 'Gemini', 'Anthropic'.
        api_key: Required for cloud providers.
        ollama_url: Base URL for Ollama.

    Returns:
        A concrete LLMProvider.

    Raises:
        ProviderAuthError: If a cloud provider is requested without an API key.
        ValueError: If the provider name is unknown.
    """
    if name == "Ollama":
        return OllamaProvider(base_url=ollama_url)
    if name == "OpenAI":
        if not api_key:
            raise ProviderAuthError("OpenAI requires an API key.")
        return OpenAIProvider(api_key=api_key)
    if name == "Gemini":
        if not api_key:
            raise ProviderAuthError("Gemini requires an API key.")
        return GeminiProvider(api_key=api_key)
    if name == "Anthropic":
        if not api_key:
            raise ProviderAuthError("Anthropic requires an API key.")
        return AnthropicProvider(api_key=api_key)
    raise ValueError(f"Unknown provider: {name}")
