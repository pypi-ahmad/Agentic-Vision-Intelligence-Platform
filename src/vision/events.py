"""Event extraction from consecutive CV detection results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SceneEvent:
    """A notable event extracted from CV outputs."""
    event_type: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    frame_index: int = 0
    involved_objects: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "info"  # info | warning | alert
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "frame_index": self.frame_index,
            "severity": self.severity,
            "involved_objects": self.involved_objects,
            "metadata": self.metadata,
        }


class EventExtractor:
    """Compare consecutive FrameResults and emit events with cooldown."""

    def __init__(self, cooldown_seconds: int = 10):
        self._cooldown = cooldown_seconds
        self._prev_counts: dict[str, int] = {}
        self._prev_tracks: set[int] = set()
        self._last_emit: dict[str, float] = {}

    def extract(self, frame_result, frame_index: int = 0) -> list[SceneEvent]:
        events: list[SceneEvent] = []
        now = datetime.now()
        counts = frame_result.object_counts
        cur_tracks: set[int] = {d.track_id for d in frame_result.detections if d.track_id is not None}

        # new class appeared
        for cls, cnt in counts.items():
            if cls not in self._prev_counts and self._ok("appeared:" + cls):
                events.append(SceneEvent("object_appeared", f"{cnt} '{cls}' appeared", now, frame_index, severity="info",
                                         involved_objects=[{"class": cls, "count": cnt}]))

        # class left
        for cls in self._prev_counts:
            if cls not in counts and self._ok("left:" + cls):
                events.append(SceneEvent("object_left", f"'{cls}' left the scene", now, frame_index, severity="info",
                                         involved_objects=[{"class": cls}]))

        # significant count change
        for cls, cnt in counts.items():
            prev = self._prev_counts.get(cls, 0)
            if prev > 0 and abs(cnt - prev) >= max(2, prev * 0.5):
                if self._ok("count:" + cls):
                    direction = "increased" if cnt > prev else "decreased"
                    events.append(SceneEvent("count_change", f"'{cls}' {direction}: {prev}→{cnt}", now, frame_index,
                                             severity="warning" if cnt > prev else "info",
                                             involved_objects=[{"class": cls, "prev": prev, "current": cnt}]))

        # crowding
        for cls, cnt in counts.items():
            if cnt >= 10 and self._ok("crowd:" + cls):
                events.append(SceneEvent("crowding", f"High density of '{cls}': {cnt}", now, frame_index,
                                         severity="warning", involved_objects=[{"class": cls, "count": cnt}]))

        # new tracked objects (batch)
        new_t = cur_tracks - self._prev_tracks
        if new_t and len(new_t) >= 3 and self._ok("new_tracks"):
            events.append(SceneEvent("new_tracks", f"{len(new_t)} new tracked objects entered", now, frame_index,
                                     severity="info", metadata={"track_ids": sorted(new_t)}))

        self._prev_counts = dict(counts)
        self._prev_tracks = cur_tracks
        return events

    def _ok(self, key: str) -> bool:
        now = time.time()
        if now - self._last_emit.get(key, 0.0) < self._cooldown:
            return False
        self._last_emit[key] = now
        return True

    def reset(self) -> None:
        self._prev_counts.clear()
        self._prev_tracks.clear()
        self._last_emit.clear()
