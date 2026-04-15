"""LLM/VLM provider abstraction — base interface and error hierarchy."""

from __future__ import annotations

import abc
import base64
from typing import Any

import cv2
import numpy as np


# ======================================================================
# Provider error hierarchy
# ======================================================================

class ProviderError(Exception):
    """Base exception for all provider-layer errors."""


class ProviderAuthError(ProviderError):
    """Invalid, expired, or missing API key / credentials."""


class ProviderConnectionError(ProviderError):
    """Cannot reach the provider (network down, DNS failure, timeout)."""


class ProviderRateLimitError(ProviderError):
    """The provider returned a rate-limit / quota-exceeded response."""


# ======================================================================
# Abstract base
# ======================================================================

class LLMProvider(abc.ABC):
    """Abstract interface that every provider adapter must implement.

    ``list_models()`` raises a :class:`ProviderError` subclass on failure
    instead of silently returning ``[]``.  Callers that want a boolean
    check should use :meth:`is_available`.
    """

    name: str = "base"

    @abc.abstractmethod
    def list_models(self) -> list[str]:
        """Return available model names.

        Raises:
            ProviderAuthError: bad or missing credentials.
            ProviderConnectionError: network / timeout.
            ProviderRateLimitError: rate-limited.
            ProviderError: any other provider-specific failure.
        """

    @abc.abstractmethod
    def generate(self, prompt: str, *, model: str,
                 images: list[np.ndarray] | None = None,
                 system: str | None = None) -> str:
        """Generate a text completion."""

    def is_available(self) -> bool:
        """Quick reachability check (never raises)."""
        try:
            return len(self.list_models()) > 0
        except Exception:
            return False

    # --- shared helper ------------------------------------------------

    @staticmethod
    def _encode_image_b64(image: np.ndarray, quality: int = 85) -> str:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buf.tobytes()).decode()
