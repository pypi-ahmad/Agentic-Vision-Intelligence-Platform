"""YOLO26 model loading — supports n / m / l variants."""

from __future__ import annotations

import logging

from ultralytics import YOLO

from config import YOLO_MODELS

logger = logging.getLogger(__name__)

_cache: dict[str, YOLO] = {}


def load_yolo(variant: str) -> YOLO:
    """Load and cache a YOLO26 model by variant name.

    Args:
        variant: One of 'YOLO26n', 'YOLO26m', 'YOLO26l'.

    Returns:
        Loaded YOLO model (auto-downloads weights on first use).
    """
    if variant in _cache:
        return _cache[variant]
    weight = YOLO_MODELS.get(variant)
    if weight is None:
        raise ValueError(f"Unknown YOLO variant '{variant}'. Choose from {list(YOLO_MODELS)}")
    logger.info("Loading %s (%s)…", variant, weight)
    model = YOLO(weight)
    _cache[variant] = model
    return model
