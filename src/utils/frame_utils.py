"""Frame-level utilities — resize, convert, annotate."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def resize_frame(frame: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
