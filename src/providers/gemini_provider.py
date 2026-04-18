"""Google Gemini provider — cloud, requires API key."""

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


def _classify(exc: Exception) -> type[ProviderError]:
    msg = str(exc).lower()
    if "api key" in msg or "api_key" in msg or "401" in msg or "403" in msg or "invalid" in msg:
        return ProviderAuthError
    if "quota" in msg or "429" in msg or "rate" in msg:
        return ProviderRateLimitError
    if "connect" in msg or "timeout" in msg or "resolve" in msg or "network" in msg:
        return ProviderConnectionError
    return ProviderError


class GeminiProvider(LLMProvider):
    """``google-genai`` SDK adapter.

    * Caches a single ``genai.Client`` on the instance.
    * Sends images as documented ``types.Part.from_bytes`` parts rather than
      raw PIL objects (the PIL shortcut is not in the official docs and is
      best avoided).
    * Passes system instructions through ``GenerateContentConfig`` rather
      than concatenating into the prompt string.
    """

    name = "Gemini"

    def __init__(self, api_key: str):
        self._key = api_key
        self._client_instance: Any = None

    def _client(self):
        if self._client_instance is None:
            from google import genai
            self._client_instance = genai.Client(api_key=self._key)
        return self._client_instance

    def list_models(self) -> list[str]:
        try:
            client = self._client()
            models = client.models.list()
            return sorted(
                (m.name or "").removeprefix("models/")
                for m in models
                if "gemini" in (m.name or "").lower()
            )
        except Exception as exc:
            raise _classify(exc)(
                {
                    ProviderAuthError: "Invalid Gemini API key. Check your key at aistudio.google.com.",
                    ProviderRateLimitError: "Gemini rate limit / quota exceeded. Wait and retry.",
                    ProviderConnectionError: "Cannot reach Google Gemini API — check your network.",
                    ProviderError: f"Gemini model listing failed: {exc}",
                }[_classify(exc)]
            ) from exc

    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        from google.genai import types as _types

        client = self._client()
        contents: list[Any] = []
        if images:
            for img in images:
                jpeg = self._encode_image_jpeg(img)
                contents.append(_types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"))
        contents.append(prompt)

        config = None
        if system:
            config = _types.GenerateContentConfig(system_instruction=system)

        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            err_cls = _classify(exc)
            msg = {
                ProviderAuthError: "Invalid Gemini API key.",
                ProviderRateLimitError: "Gemini rate limit exceeded.",
                ProviderConnectionError: "Cannot reach Gemini API.",
                ProviderError: f"Gemini generation failed: {exc}",
            }[err_cls]
            raise err_cls(msg) from exc
        return resp.text or ""
