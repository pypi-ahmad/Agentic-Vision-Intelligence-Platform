"""Tests for session persistence."""

import json

from src.memory.session_store import SessionStore


class TestSessionStore:
    def test_creates_session_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.memory.session_store.get_settings", lambda: type("S", (), {"output_path": tmp_path})())
        store = SessionStore(session_id="abc123")
        assert store.path.exists()
        assert store.session_id == "abc123"

    def test_save_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.memory.session_store.get_settings", lambda: type("S", (), {"output_path": tmp_path})())
        store = SessionStore(session_id="abc123")
        fp = store.save_json("data.json", {"ok": True})
        assert json.loads(fp.read_text(encoding="utf-8"))["ok"] is True

    def test_save_text(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.memory.session_store.get_settings", lambda: type("S", (), {"output_path": tmp_path})())
        store = SessionStore(session_id="abc123")
        fp = store.save_text("notes.txt", "hello")
        assert fp.read_text(encoding="utf-8") == "hello"
