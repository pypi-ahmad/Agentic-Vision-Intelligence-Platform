"""Google Gemini provider — cloud, requires API key."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "Gemini"

    def __init__(self, api_key: str):
        self._key = api_key

    def _client(self):
        from google import genai
        return genai.Client(api_key=self._key)

    def list_models(self) -> list[str]:
        try:
            client = self._client()
            models = client.models.list()
            return sorted(
                m.name.replace("models/", "")
                for m in models
                if "gemini" in (m.name or "").lower()
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "api key" in msg or "api_key" in msg or "invalid" in msg or "401" in msg or "403" in msg:
                raise ProviderAuthError(
                    "Invalid Gemini API key. Check your key at aistudio.google.com."
                ) from exc
            if "quota" in msg or "429" in msg or "rate" in msg:
                raise ProviderRateLimitError(
                    "Gemini rate limit / quota exceeded. Wait and retry."
                ) from exc
            if "connect" in msg or "timeout" in msg or "resolve" in msg or "network" in msg:
                raise ProviderConnectionError(
                    "Cannot reach Google Gemini API — check your network."
                ) from exc
            raise ProviderError(f"Gemini model listing failed: {exc}") from exc

    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        client = self._client()
        contents: list[Any] = []
        if system:
            prompt = f"{system}\n\n{prompt}"
        if images:
            for img in images:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                contents.append(pil)
        contents.append(prompt)
        try:
            resp = client.models.generate_content(model=model, contents=contents)
        except Exception as exc:
            msg = str(exc).lower()
            if "api key" in msg or "api_key" in msg or "invalid" in msg or "401" in msg or "403" in msg:
                raise ProviderAuthError("Invalid Gemini API key.") from exc
            if "quota" in msg or "429" in msg or "rate" in msg:
                raise ProviderRateLimitError("Gemini rate limit exceeded.") from exc
            if "connect" in msg or "timeout" in msg or "resolve" in msg or "network" in msg:
                raise ProviderConnectionError("Cannot reach Gemini API.") from exc
            raise ProviderError(f"Gemini generation failed: {exc}") from exc
        return resp.text or ""
