"""OpenAI provider — cloud, requires API key."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)

logger = logging.getLogger(__name__)

_SKIP = frozenset({
    "embedding", "tts", "whisper", "dall-e",
    "moderation", "davinci", "babbage", "search",
})


class OpenAIProvider(LLMProvider):
    name = "OpenAI"

    def __init__(self, api_key: str):
        self._key = api_key

    def _client(self):
        import openai
        return openai.OpenAI(api_key=self._key)

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
                f"OpenAI API returned status {exc.status_code}: {exc.message}"
            ) from exc
        except Exception as exc:
            raise ProviderError(f"OpenAI model listing failed: {exc}") from exc

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

        try:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=2048)
        except _openai.AuthenticationError as exc:
            raise ProviderAuthError("Invalid OpenAI API key.") from exc
        except _openai.RateLimitError as exc:
            raise ProviderRateLimitError("OpenAI rate limit exceeded.") from exc
        except (_openai.APIConnectionError, _openai.APITimeoutError) as exc:
            raise ProviderConnectionError("Cannot reach OpenAI API.") from exc
        except _openai.APIStatusError as exc:
            raise ProviderError(f"OpenAI API error (HTTP {exc.status_code})") from exc
        return resp.choices[0].message.content or ""
