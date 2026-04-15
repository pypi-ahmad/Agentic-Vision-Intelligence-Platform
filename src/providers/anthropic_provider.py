"""Anthropic provider — cloud, requires API key."""

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


class AnthropicProvider(LLMProvider):
    name = "Anthropic"

    def __init__(self, api_key: str):
        self._key = api_key

    def _client(self):
        import anthropic
        return anthropic.Anthropic(api_key=self._key)

    def list_models(self) -> list[str]:
        import anthropic as _anthropic

        try:
            client = self._client()
            resp = client.models.list(limit=50)
            return sorted(m.id for m in resp.data)
        except _anthropic.AuthenticationError as exc:
            raise ProviderAuthError(
                "Invalid Anthropic API key. Check your key at console.anthropic.com."
            ) from exc
        except _anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(
                "Anthropic rate limit exceeded. Wait a moment and retry."
            ) from exc
        except (_anthropic.APIConnectionError, _anthropic.APITimeoutError) as exc:
            raise ProviderConnectionError(
                "Cannot reach Anthropic API — check your network."
            ) from exc
        except _anthropic.APIStatusError as exc:
            raise ProviderError(
                f"Anthropic API error (HTTP {exc.status_code}): {exc.message}"
            ) from exc
        except Exception as exc:
            raise ProviderError(f"Anthropic model listing failed: {exc}") from exc

    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        client = self._client()
        content: list[dict[str, Any]] = []

        if images:
            for img in images:
                b64 = self._encode_image_b64(img)
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
        content.append({"type": "text", "text": prompt})

        kwargs: dict[str, Any] = {"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": content}]}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text if resp.content else ""
