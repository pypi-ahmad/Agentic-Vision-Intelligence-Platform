"""Live camera / webcam input source."""

from __future__ import annotations

from datetime import datetime

import cv2

from config import get_settings
from src.input.base import FramePacket, InputSource


class CameraSource(InputSource):
    source_type = "camera"

    def __init__(self, camera_index: int | None = None):
        cfg = get_settings()
        self._index = camera_index if camera_index is not None else cfg.camera_index
        self._width = cfg.camera_width
        self._height = cfg.camera_height
        self._cap: cv2.VideoCapture | None = None
        self._n = 0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self._index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._n = 0

    def read(self) -> FramePacket | None:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._n += 1
        return FramePacket(
            frame=frame,
            timestamp=datetime.now(),
            frame_index=self._n,
            source_id=f"camera:{self._index}",
            source_type=self.source_type,
            metadata={"width": frame.shape[1], "height": frame.shape[0]},
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_live(self) -> bool:
        return True
