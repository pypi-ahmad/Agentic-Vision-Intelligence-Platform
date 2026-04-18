"""Session persistence — save / load session data as JSON."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import get_settings

# Session identifiers are interpolated into a filesystem path; reject
# anything that could traverse or reference an unexpected directory.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
# Filenames saved under the session directory must stay flat — no separators,
# no parent references, just a plain name with optional extension.
_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-.]{1,128}$")


def _validate_session_id(sid: str) -> str:
    if not _SESSION_ID_RE.fullmatch(sid):
        raise ValueError(
            "Invalid session_id: must match [A-Za-z0-9_-]{1,64}."
        )
    return sid


def _validate_filename(name: str) -> str:
    if not _FILENAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(
            f"Invalid filename {name!r}: must match [A-Za-z0-9_.-]{{1,128}} and contain no traversal."
        )
    return name


class SessionStore:
    def __init__(self, session_id: str | None = None):
        self._sid = _validate_session_id(session_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
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
        fp = self._dir / _validate_filename(filename)
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        return fp

    def save_text(self, filename: str, text: str) -> Path:
        fp = self._dir / _validate_filename(filename)
        fp.write_text(text, encoding="utf-8")
        return fp
