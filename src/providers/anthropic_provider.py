"""Anthropic provider — cloud, requires API key."""

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

_MAX_TOKENS = 2048


class AnthropicProvider(LLMProvider):
    """Anthropic SDK adapter.

    * Caches the ``anthropic.Anthropic`` client on the instance — one HTTP
      transport per provider rather than per call.
    * Wraps the system prompt in a cache-controlled content block when one is
      supplied, opting into Anthropic's automatic prompt caching.  Repeat
      calls within the cache TTL re-use the same prefix and avoid re-paying
      for the system instruction's tokens.
    """

    name = "Anthropic"

    def __init__(self, api_key: str):
        self._key = api_key
        self._client_instance: Any = None

    def _client(self):
        if self._client_instance is None:
            import anthropic
            self._client_instance = anthropic.Anthropic(api_key=self._key)
        return self._client_instance

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
                f"Anthropic API error (HTTP {exc.status_code})"
            ) from exc
        except Exception as exc:
            raise ProviderError("Anthropic model listing failed.") from exc

    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        import anthropic as _anthropic

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

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            # Prompt caching: wrap the stable system instruction as a text
            # block with ephemeral cache_control so repeat requests in the
            # cache TTL skip re-processing the prefix.
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        try:
            resp = client.messages.create(**kwargs)
        except _anthropic.AuthenticationError as exc:
            raise ProviderAuthError("Invalid Anthropic API key.") from exc
        except _anthropic.RateLimitError as exc:
            raise ProviderRateLimitError("Anthropic rate limit exceeded.") from exc
        except (_anthropic.APIConnectionError, _anthropic.APITimeoutError) as exc:
            raise ProviderConnectionError("Cannot reach Anthropic API.") from exc
        except _anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error (HTTP {exc.status_code})") from exc
        return resp.content[0].text if resp.content else ""
