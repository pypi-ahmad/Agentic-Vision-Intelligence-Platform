"""YOLO26 detection and tracking engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import get_settings
from src.vision.models import load_yolo


@dataclass
class Detection:
    """Single detection from one frame."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1 y1 x2 y2
    track_id: int | None = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
        }
        if self.track_id is not None:
            d["track_id"] = self.track_id
        return d


@dataclass
class FrameResult:
    """Aggregated result for one frame."""
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    object_counts: dict[str, int] = field(default_factory=dict)
    annotated_frame: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "num_detections": len(self.detections),
            "object_counts": self.object_counts,
            "detections": [d.to_dict() for d in self.detections],
        }

    @property
    def summary_line(self) -> str:
        parts = [f"{v} {k}" for k, v in sorted(self.object_counts.items(), key=lambda x: -x[1])]
        return ", ".join(parts) if parts else "no objects detected"


class VisionDetector:
    """Unified YOLO26 inference interface that handles any variant."""

    def __init__(self, variant: str = "YOLO26n", confidence: float | None = None):
        cfg = get_settings()
        self._variant = variant
        self._conf = confidence if confidence is not None else cfg.yolo_confidence
        self._device = cfg.resolve_device()
        self._model = load_yolo(variant)

    @property
    def variant(self) -> str:
        return self._variant

    # ---- detection ---------------------------------------------------

    def detect(self, frame: np.ndarray, *, confidence: float | None = None) -> FrameResult:
        conf = confidence if confidence is not None else self._conf
        results = self._model.predict(
            source=frame, conf=conf,
            device=self._device, verbose=False,
        )
        return self._parse(results)

    # ---- tracking ----------------------------------------------------

    def track(self, frame: np.ndarray, *, confidence: float | None = None, persist: bool = True) -> FrameResult:
        conf = confidence if confidence is not None else self._conf
        results = self._model.track(
            source=frame, conf=conf,
            device=self._device, persist=persist, verbose=False,
        )
        return self._parse(results)

    # ---- internal ----------------------------------------------------

    def _parse(self, results) -> FrameResult:
        dets: list[Detection] = []
        counts: dict[str, int] = {}
        if not results:
            return FrameResult(frame_index=0)
        r = results[0]
        boxes = r.boxes
        names = r.names
        annotated = r.plot()
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cid = int(boxes.cls[i].item())
                cname = names.get(cid, f"class_{cid}")
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                tid = int(boxes.id[i].item()) if boxes.id is not None else None
                dets.append(Detection(class_id=cid, class_name=cname, confidence=conf,
                                      bbox=(int(x1), int(y1), int(x2), int(y2)), track_id=tid))
                counts[cname] = counts.get(cname, 0) + 1
        return FrameResult(frame_index=0, detections=dets, object_counts=counts, annotated_frame=annotated)
