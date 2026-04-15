"""Image file / in-memory image input."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.input.base import FramePacket, InputSource


class ImageSource(InputSource):
    """Single image file input."""
    source_type = "image"

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._frame: np.ndarray | None = None
        self._consumed = False

    def open(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Image not found: {self._path}")
        img = cv2.imread(str(self._path))
        if img is None:
            raise ValueError(f"Cannot decode image: {self._path}")
        self._frame = img
        self._consumed = False

    def read(self) -> FramePacket | None:
        if self._consumed or self._frame is None:
            return None
        self._consumed = True
        return FramePacket(
            frame=self._frame, timestamp=datetime.now(), frame_index=0,
            source_id=str(self._path), source_type=self.source_type,
            metadata={"filename": self._path.name, "width": self._frame.shape[1], "height": self._frame.shape[0]},
        )

    def close(self) -> None:
        self._frame = None
        self._consumed = False


class ImageArraySource(InputSource):
    """In-memory numpy array as a single-frame source."""
    source_type = "image"

    def __init__(self, image: np.ndarray, name: str = "uploaded"):
        self._image = image
        self._name = name
        self._consumed = False

    def open(self) -> None:
        self._consumed = False

    def read(self) -> FramePacket | None:
        if self._consumed:
            return None
        self._consumed = True
        return FramePacket(
            frame=self._image, timestamp=datetime.now(), frame_index=0,
            source_id=self._name, source_type=self.source_type,
            metadata={"width": self._image.shape[1], "height": self._image.shape[0]},
        )

    def close(self) -> None:
        self._consumed = False
