"""Session export — save reports, event logs, summaries to disk."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


class SessionExporter:
    """Export session artefacts to the output directory."""

    def __init__(self, session_id: str | None = None):
        self._sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = get_settings()
        self._dir = cfg.output_path / f"session_{self._sid}"
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_id(self) -> str:
        return self._sid

    @property
    def export_dir(self) -> Path:
        return self._dir

    def save_report(self, report_md: str) -> Path:
        fp = self._dir / "report.md"
        fp.write_text(report_md, encoding="utf-8")
        logger.info("Report saved → %s", fp)
        return fp

    def save_events(self, events: list[dict[str, Any]]) -> Path:
        fp = self._dir / "events.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, default=str, ensure_ascii=False)
        return fp

    def save_alerts(self, alerts: list[dict[str, Any]]) -> Path:
        fp = self._dir / "alerts.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2, default=str, ensure_ascii=False)
        return fp

    def save_summary(self, summary: dict[str, Any]) -> Path:
        fp = self._dir / "summary.json"
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        return fp

    def save_text(self, filename: str, text: str) -> Path:
        fp = self._dir / filename
        fp.write_text(text, encoding="utf-8")
        return fp
