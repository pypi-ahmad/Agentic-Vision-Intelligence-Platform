"""Ollama provider — local-first, no API key needed."""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np

from src.providers.base import LLMProvider, ProviderConnectionError, ProviderError


class OllamaProvider(LLMProvider):
    name = "Ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._url = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self._url}/api/tags", timeout=8)
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
            raise ProviderConnectionError(
                f"Cannot reach Ollama at {self._url} — is it running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
        except Exception as exc:
            raise ProviderError(f"Ollama model listing failed: {exc}") from exc

    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if images:
            payload["images"] = [self._encode_image_b64(img) for img in images]
        try:
            r = httpx.post(f"{self._url}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException) as exc:
            raise ProviderConnectionError(
                f"Cannot reach Ollama at {self._url} — is it running?"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"Ollama returned HTTP {exc.response.status_code}"
            ) from exc
        body = r.json()
        if "error" in body:
            raise ProviderError(f"Ollama error: {body['error']}")
        return body.get("response", "")

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self._url}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False
