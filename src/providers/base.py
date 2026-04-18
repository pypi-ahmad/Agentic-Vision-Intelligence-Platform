"""LLM/VLM provider abstraction — base interface and error hierarchy."""

from __future__ import annotations

import abc
import base64
import hashlib
from functools import lru_cache

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

    # --- shared helpers -----------------------------------------------

    @staticmethod
    def _encode_image_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
        """Encode a BGR frame as JPEG bytes (pre-base64).

        Result is memoised by ``(image-bytes hash, quality)`` so repeat calls
        on the same frame in a single perception cycle (e.g. describe + anomaly
        reasoning) don't re-encode.
        """
        key = (hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest(), quality)
        return _cached_jpeg_encode(key, image.shape, image.dtype.str, image.tobytes(), quality)

    @staticmethod
    def _encode_image_b64(image: np.ndarray, quality: int = 85) -> str:
        return base64.b64encode(LLMProvider._encode_image_jpeg(image, quality)).decode()


@lru_cache(maxsize=32)
def _cached_jpeg_encode(
    _key: tuple[str, int],
    shape: tuple,
    dtype_str: str,
    raw: bytes,
    quality: int,
) -> bytes:
    """Backing store for :meth:`LLMProvider._encode_image_jpeg`.

    Keyed on a BLAKE2b hash of the frame bytes.  ``shape``/``dtype_str``/``raw``
    are reconstructed into a numpy array for ``cv2.imencode``.  LRU-capped at
    32 entries (~a few MB) to bound memory.
    """
    arr = np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode image")
    return buf.tobytes()
