"""Video file input source."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from src.input.base import FramePacket, InputSource


class VideoSource(InputSource):
    source_type = "video"

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._n = 0
        self._fps: float = 0.0
        self._total: int = 0

    def open(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self._path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._n = 0

    def read(self) -> FramePacket | None:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        idx = self._n
        self._n += 1
        t = idx / self._fps if self._fps > 0 else 0.0
        return FramePacket(
            frame=frame, timestamp=datetime.now(), frame_index=idx,
            source_id=str(self._path), source_type=self.source_type,
            metadata={"filename": self._path.name, "fps": self._fps,
                       "total_frames": self._total, "video_time_sec": round(t, 3),
                       "width": frame.shape[1], "height": frame.shape[0]},
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def duration_seconds(self) -> float:
        return self._total / self._fps if self._fps > 0 and self._total > 0 else 0.0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total
