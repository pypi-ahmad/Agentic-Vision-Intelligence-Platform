"""Current scene state tracking — rolling window of tracked objects."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    frame_count: int = 0

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    def to_dict(self) -> dict[str, Any]:
        return {"track_id": self.track_id, "class_name": self.class_name,
                "duration_sec": round(self.duration, 1), "frame_count": self.frame_count}


class SceneState:
    """Maintains a rolling window of tracked objects and counts."""

    def __init__(self, window_seconds: int = 300):
        self._window = window_seconds
        self._tracked: dict[int, TrackedObject] = {}
        self._counts: dict[str, int] = {}
        self._total_frames = 0

    def update(self, frame_result) -> None:
        now = time.time()
        self._total_frames += 1
        self._counts = dict(frame_result.object_counts)
        for det in frame_result.detections:
            if det.track_id is None:
                continue
            if det.track_id in self._tracked:
                o = self._tracked[det.track_id]
                o.last_seen = now
                o.last_bbox = det.bbox
                o.frame_count += 1
            else:
                self._tracked[det.track_id] = TrackedObject(
                    track_id=det.track_id, class_name=det.class_name,
                    first_seen=now, last_seen=now, last_bbox=det.bbox, frame_count=1)
        # evict
        cutoff = now - self._window
        self._tracked = {k: v for k, v in self._tracked.items() if v.last_seen >= cutoff}

    @property
    def active_objects(self) -> list[TrackedObject]:
        now = time.time()
        return [o for o in self._tracked.values() if now - o.last_seen < 5.0]

    @property
    def all_tracked(self) -> list[TrackedObject]:
        return list(self._tracked.values())

    @property
    def current_counts(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def get_summary(self) -> dict[str, Any]:
        active = self.active_objects
        cls_counts: dict[str, int] = defaultdict(int)
        for o in active:
            cls_counts[o.class_name] += 1
        return {
            "timestamp": datetime.now().isoformat(),
            "total_frames": self._total_frames,
            "current_counts": self._counts,
            "active_tracked": len(active),
            "tracked_classes": dict(cls_counts),
            "total_in_memory": len(self._tracked),
        }

    def get_description(self) -> str:
        if not self._counts:
            return "The scene is empty — no objects detected."
        parts = [f"{v} {k}" for k, v in sorted(self._counts.items(), key=lambda x: -x[1])]
        desc = f"Current scene: {', '.join(parts)}."
        a = len(self.active_objects)
        if a:
            desc += f" {a} object(s) actively tracked."
        return desc

    def reset(self) -> None:
        self._tracked.clear()
        self._counts.clear()
        self._total_frames = 0
