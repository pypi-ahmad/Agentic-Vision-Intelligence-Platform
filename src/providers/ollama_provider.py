"""Ollama provider — local-first, no API key needed."""

from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np

from config import get_settings
from src.providers.base import (
    LLMProvider,
    ProviderConnectionError,
    ProviderError,
)


def _validate_ollama_url(base_url: str) -> str:
    """Reject Ollama base URLs that could enable SSRF.

    When ``allow_remote_ollama`` is ``False`` (the default), the host must be
    loopback or an RFC-1918 private address.  Raises :class:`ValueError` for
    malformed URLs, public addresses, or unsupported schemes.
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Ollama URL must be http(s): got {base_url!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"Ollama URL has no host: {base_url!r}")

    cfg = get_settings()
    if cfg.allow_remote_ollama:
        return base_url

    # Accept common local hostnames outright.
    if host.lower() in {"localhost", "ollama"}:
        return base_url

    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            f"Ollama URL {base_url!r} is not a private host. "
            "Set ALLOW_REMOTE_OLLAMA=1 to permit remote daemons."
        ) from exc

    # ``is_link_local`` intentionally excluded: 169.254.0.0/16 is the cloud
    # instance-metadata range (AWS/GCP/Azure), a prime SSRF target.
    if ip.is_loopback or (ip.is_private and not ip.is_link_local):
        return base_url
    raise ValueError(
        f"Ollama URL {base_url!r} points at a non-private/link-local address. "
        "Set ALLOW_REMOTE_OLLAMA=1 to permit remote daemons."
    )


class OllamaProvider(LLMProvider):
    """Local Ollama REST adapter.

    Uses a single ``httpx.Client`` per provider instance so repeat requests to
    the local daemon reuse the same TCP connection instead of reconnecting per
    call (the module-level ``httpx.get``/``post`` functions do **not** pool).
    """

    name = "Ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._url = _validate_ollama_url(base_url.rstrip("/"))
        # Fresh client per provider — closed on __del__.  Timeouts split: short
        # connect so a wrong URL surfaces fast, long read for LLM generation.
        self._client: httpx.Client = httpx.Client(
            base_url=self._url,
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    def __del__(self) -> None:
        try:
            client = getattr(self, "_client", None)
            if client is not None:
                client.close()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    # ---- API ---------------------------------------------------------

    def list_models(self) -> list[str]:
        try:
            r = self._client.get("/api/tags", timeout=8.0)
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
            raise ProviderError("Ollama model listing failed.") from exc

    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        if images:
            payload["images"] = [self._encode_image_b64(img) for img in images]
        try:
            r = self._client.post("/api/generate", json=payload)
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
            r = self._client.get("/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False
