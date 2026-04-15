"""Session persistence — save / load session data as JSON."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_settings

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, session_id: str | None = None):
        self._sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = get_settings()
        self._dir = cfg.output_path / f"session_{self._sid}"
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_id(self) -> str:
        return self._sid

    @property
    def path(self) -> Path:
        return self._dir

    def save_json(self, filename: str, data: Any) -> Path:
        fp = self._dir / filename
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        return fp

    def save_text(self, filename: str, text: str) -> Path:
        fp = self._dir / filename
        fp.write_text(text, encoding="utf-8")
        return fp
