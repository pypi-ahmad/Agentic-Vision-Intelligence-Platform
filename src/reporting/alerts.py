"""Alert management — collect, filter, and format alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Alert:
    alert_id: int
    event_type: str
    severity: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    frame_index: int = 0
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "frame_index": self.frame_index,
            "acknowledged": self.acknowledged,
        }


class AlertManager:
    """Collect and prioritise alerts from scene events."""

    def __init__(self):
        self._alerts: list[Alert] = []
        self._next_id = 1

    def ingest_events(self, events: list[dict[str, Any]]) -> list[Alert]:
        new: list[Alert] = []
        for ev in events:
            if ev.get("severity") in ("warning", "alert"):
                a = Alert(
                    alert_id=self._next_id,
                    event_type=ev.get("event_type", "unknown"),
                    severity=ev.get("severity", "warning"),
                    description=ev.get("description", ""),
                    frame_index=ev.get("frame_index", 0),
                )
                self._alerts.append(a)
                new.append(a)
                self._next_id += 1
        return new

    def acknowledge(self, alert_id: int) -> bool:
        for a in self._alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

    @property
    def unacknowledged(self) -> list[Alert]:
        return [a for a in self._alerts if not a.acknowledged]

    @property
    def all_alerts(self) -> list[Alert]:
        return list(self._alerts)

    def to_list(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._alerts]

    def reset(self) -> None:
        self._alerts.clear()
        self._next_id = 1
