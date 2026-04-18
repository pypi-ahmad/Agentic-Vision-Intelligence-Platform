"""Security tests — SSRF allowlist for Ollama URL, session_id path traversal."""

from __future__ import annotations

import pytest

from src.memory.session_store import SessionStore, _validate_filename, _validate_session_id
from src.providers.ollama_provider import OllamaProvider, _validate_ollama_url
from src.reporting.exporter import SessionExporter


class TestOllamaUrlValidation:
    """S2: prevent SSRF via the ``Ollama URL`` sidebar text input."""

    def test_localhost_allowed(self):
        assert _validate_ollama_url("http://localhost:11434") == "http://localhost:11434"

    def test_loopback_ipv4_allowed(self):
        assert _validate_ollama_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"

    def test_private_rfc1918_allowed(self):
        assert _validate_ollama_url("http://10.0.0.5:11434") == "http://10.0.0.5:11434"
        assert _validate_ollama_url("http://192.168.1.50:11434") == "http://192.168.1.50:11434"

    def test_public_ip_rejected(self):
        with pytest.raises(ValueError, match="non-private"):
            _validate_ollama_url("http://8.8.8.8:11434")

    def test_metadata_service_rejected(self):
        """AWS / GCP metadata endpoint is a classic SSRF target — must be blocked."""
        with pytest.raises(ValueError):
            _validate_ollama_url("http://169.254.169.254/latest/meta-data/")

    def test_arbitrary_hostname_rejected(self):
        with pytest.raises(ValueError, match="private"):
            _validate_ollama_url("http://attacker.example.com:11434")

    def test_wrong_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            _validate_ollama_url("file:///etc/passwd")

    def test_allow_remote_override(self, monkeypatch):
        """Operator opt-in via ALLOW_REMOTE_OLLAMA must unblock public hosts."""
        import config

        real_settings = config.get_settings()
        monkeypatch.setattr(real_settings, "allow_remote_ollama", True)
        # Operator explicitly allowed remote — validation becomes a no-op.
        assert _validate_ollama_url("http://attacker.example.com:11434") == "http://attacker.example.com:11434"

    def test_constructor_rejects_public_host(self):
        with pytest.raises(ValueError):
            OllamaProvider(base_url="http://8.8.8.8:11434")


class TestSessionIdValidation:
    """S4: reject session IDs that could traverse the filesystem."""

    def test_valid_id(self):
        assert _validate_session_id("20260418_120000") == "20260418_120000"
        assert _validate_session_id("abc-123_XYZ") == "abc-123_XYZ"

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            _validate_session_id("")

    def test_traversal_rejected(self):
        for attempt in ("..", "../evil", "a/b", "a\\b", "a:b", "."):
            with pytest.raises(ValueError):
                _validate_session_id(attempt)

    def test_too_long_rejected(self):
        with pytest.raises(ValueError):
            _validate_session_id("x" * 65)

    def test_filename_rejects_traversal(self):
        with pytest.raises(ValueError):
            _validate_filename("../escape.json")
        with pytest.raises(ValueError):
            _validate_filename("nested/dir/file.txt")

    def test_filename_accepts_normal_names(self):
        assert _validate_filename("events.json") == "events.json"
        assert _validate_filename("chat_history-v2.json") == "chat_history-v2.json"

    def test_session_store_rejects_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.memory.session_store.get_settings",
            lambda: type("S", (), {"output_path": tmp_path})(),
        )
        with pytest.raises(ValueError):
            SessionStore(session_id="../../../etc")

    def test_session_exporter_rejects_traversal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.reporting.exporter.get_settings",
            lambda: type("S", (), {"output_path": tmp_path})(),
        )
        with pytest.raises(ValueError):
            SessionExporter(session_id="..\\..")
