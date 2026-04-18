"""Tests for reporting — AlertManager, SessionExporter."""



from src.reporting.alerts import AlertManager
from src.reporting.exporter import SessionExporter


class TestAlertManager:
    def test_ingest_warning(self):
        am = AlertManager()
        evts = [{"event_type": "crowding", "severity": "warning", "description": "too many"}]
        new = am.ingest_events(evts)
        assert len(new) == 1
        assert new[0].severity == "warning"

    def test_info_ignored(self):
        am = AlertManager()
        evts = [{"event_type": "appeared", "severity": "info", "description": "ok"}]
        new = am.ingest_events(evts)
        assert len(new) == 0

    def test_acknowledge(self):
        am = AlertManager()
        am.ingest_events([{"event_type": "t", "severity": "alert", "description": "x"}])
        assert len(am.unacknowledged) == 1
        am.acknowledge(1)
        assert len(am.unacknowledged) == 0

    def test_reset(self):
        am = AlertManager()
        am.ingest_events([{"event_type": "t", "severity": "alert", "description": "x"}])
        am.reset()
        assert len(am.all_alerts) == 0

    def test_to_list(self):
        am = AlertManager()
        am.ingest_events([{"event_type": "t", "severity": "warning", "description": "d"}])
        lst = am.to_list()
        assert len(lst) == 1
        assert "alert_id" in lst[0]


class TestSessionExporter:
    def test_export_creates_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.reporting.exporter.get_settings",
                            lambda: type("S", (), {"output_path": tmp_path})())
        exp = SessionExporter(session_id="test1")
        assert exp.export_dir.exists()

    def test_save_text(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.reporting.exporter.get_settings",
                            lambda: type("S", (), {"output_path": tmp_path})())
        exp = SessionExporter(session_id="test2")
        fp = exp.save_text("notes.txt", "hello")
        assert fp.read_text() == "hello"

    def test_save_report(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.reporting.exporter.get_settings",
                            lambda: type("S", (), {"output_path": tmp_path})())
        exp = SessionExporter(session_id="test3")
        fp = exp.save_report("# Report")
        assert fp.read_text() == "# Report"
