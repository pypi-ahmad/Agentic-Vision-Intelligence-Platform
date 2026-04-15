"""Queryable event timeline."""

from __future__ import annotations

from typing import Any

from src.vision.events import SceneEvent


class EventTimeline:
    def __init__(self, max_events: int = 1000):
        self._events: list[SceneEvent] = []
        self._max = max_events

    def add(self, event: SceneEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max:
            self._events = self._events[-self._max:]

    def add_many(self, events: list[SceneEvent]) -> None:
        for e in events:
            self.add(e)

    @property
    def all_events(self) -> list[SceneEvent]:
        return list(self._events)

    @property
    def count(self) -> int:
        return len(self._events)

    def recent(self, n: int = 20) -> list[SceneEvent]:
        return self._events[-n:]

    def by_severity(self, severity: str) -> list[SceneEvent]:
        return [e for e in self._events if e.severity == severity]

    def warnings_and_alerts(self) -> list[SceneEvent]:
        return [e for e in self._events if e.severity in ("warning", "alert")]

    def get_summary(self) -> dict[str, Any]:
        tc: dict[str, int] = {}
        sc: dict[str, int] = {}
        for e in self._events:
            tc[e.event_type] = tc.get(e.event_type, 0) + 1
            sc[e.severity] = sc.get(e.severity, 0) + 1
        return {"total": len(self._events), "types": tc, "severities": sc,
                "recent": [e.to_dict() for e in self.recent(5)]}

    def to_text(self) -> str:
        if not self._events:
            return "No events recorded."
        return "\n".join(
            f"[{e.timestamp.strftime('%H:%M:%S')}] [{e.severity.upper()}] {e.description}"
            for e in self._events
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._events]

    def reset(self) -> None:
        self._events.clear()
