"""OpenAI provider — cloud, requires API key."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)

_SKIP = frozenset({
    "embedding", "tts", "whisper", "dall-e",
    "moderation", "davinci", "babbage", "search",
})

# Token parameter name differs between chat and reasoning model families.
# Reasoning models (o1/o3/o4/o-series, and any model beginning with "o")
# reject ``max_tokens`` and only accept ``max_completion_tokens``.
_MAX_TOKENS = 2048


def _token_param_for(model: str) -> str:
    mid = model.lower()
    # ``o1``, ``o3-mini``, ``o3``, ``o4-*`` etc. use the reasoning token name.
    if mid.startswith("o") and not mid.startswith("omni"):
        return "max_completion_tokens"
    # GPT-5/GPT-4o and later reasoning-capable chat models also moved to
    # max_completion_tokens per OpenAI migration guide.
    if mid.startswith("gpt-5"):
        return "max_completion_tokens"
    return "max_tokens"


class OpenAIProvider(LLMProvider):
    """OpenAI SDK adapter with a cached ``OpenAI`` client per provider instance."""

    name = "OpenAI"

    def __init__(self, api_key: str):
        self._key = api_key
        self._client_instance: Any = None

    # Kept as a method so existing tests that patch ``_client`` at class level
    # continue to work.  Caches the underlying SDK client on first call.
    def _client(self):
        if self._client_instance is None:
            import openai
            self._client_instance = openai.OpenAI(api_key=self._key)
        return self._client_instance

    def list_models(self) -> list[str]:
        import openai as _openai

        try:
            client = self._client()
            models = client.models.list()
            names = sorted(
                m.id for m in models
                if not any(s in m.id.lower() for s in _SKIP)
                and ("gpt" in m.id.lower() or m.id.lower().startswith("o"))
            )
            return names if names else sorted(m.id for m in models)[:20]
        except _openai.AuthenticationError as exc:
            raise ProviderAuthError(
                "Invalid OpenAI API key. Check your key at platform.openai.com."
            ) from exc
        except _openai.RateLimitError as exc:
            raise ProviderRateLimitError(
                "OpenAI rate limit exceeded. Wait a moment and retry."
            ) from exc
        except (_openai.APIConnectionError, _openai.APITimeoutError) as exc:
            raise ProviderConnectionError(
                "Cannot reach OpenAI API — check your network."
            ) from exc
        except _openai.APIStatusError as exc:
            raise ProviderError(
                f"OpenAI API returned status {exc.status_code}"
            ) from exc
        except Exception as exc:
            raise ProviderError("OpenAI model listing failed.") from exc

    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        import openai as _openai

        client = self._client()
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})

        if images:
            content: list[dict] = [{"type": "text", "text": prompt}]
            for img in images:
                b64 = self._encode_image_b64(img)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        kwargs[_token_param_for(model)] = _MAX_TOKENS

        try:
            resp = client.chat.completions.create(**kwargs)
        except _openai.AuthenticationError as exc:
            raise ProviderAuthError("Invalid OpenAI API key.") from exc
        except _openai.RateLimitError as exc:
            raise ProviderRateLimitError("OpenAI rate limit exceeded.") from exc
        except (_openai.APIConnectionError, _openai.APITimeoutError) as exc:
            raise ProviderConnectionError("Cannot reach OpenAI API.") from exc
        except _openai.APIStatusError as exc:
            raise ProviderError(f"OpenAI API error (HTTP {exc.status_code})") from exc
        return resp.choices[0].message.content or ""
