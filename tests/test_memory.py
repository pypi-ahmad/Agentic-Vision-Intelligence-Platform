"""Tests for the memory layer — SceneState, EventTimeline."""

import pytest

from src.vision.detector import Detection, FrameResult
from src.vision.events import SceneEvent
from src.memory.scene_state import SceneState
from src.memory.event_timeline import EventTimeline


class TestSceneState:
    def _frame(self, counts, dets=None, idx=0):
        dets = dets or []
        return FrameResult(frame_index=idx, detections=dets, object_counts=counts)

    def test_empty_description(self):
        ss = SceneState()
        assert "empty" in ss.get_description().lower()

    def test_update_counts(self):
        ss = SceneState()
        ss.update(self._frame({"person": 2}))
        assert ss.current_counts == {"person": 2}

    def test_total_frames(self):
        ss = SceneState()
        for i in range(5):
            ss.update(self._frame({"a": 1}, idx=i))
        assert ss.total_frames == 5

    def test_tracked_objects(self, sample_detection_tracked):
        ss = SceneState()
        ss.update(FrameResult(frame_index=0, detections=[sample_detection_tracked], object_counts={"person": 1}))
        assert len(ss.all_tracked) == 1
        assert ss.all_tracked[0].track_id == 1

    def test_reset(self):
        ss = SceneState()
        ss.update(self._frame({"x": 1}))
        ss.reset()
        assert ss.total_frames == 0
        assert ss.current_counts == {}

    def test_get_summary_keys(self):
        ss = SceneState()
        ss.update(self._frame({"person": 1}))
        s = ss.get_summary()
        for k in ("timestamp", "total_frames", "current_counts", "active_tracked"):
            assert k in s


class TestEventTimeline:
    def _event(self, desc="test", sev="info"):
        return SceneEvent(event_type="test", description=desc, severity=sev)

    def test_add_and_count(self):
        tl = EventTimeline()
        tl.add(self._event())
        assert tl.count == 1

    def test_max_events_respected(self):
        tl = EventTimeline(max_events=5)
        for i in range(10):
            tl.add(self._event(f"e{i}"))
        assert tl.count == 5

    def test_by_severity(self):
        tl = EventTimeline()
        tl.add(self._event(sev="info"))
        tl.add(self._event(sev="warning"))
        tl.add(self._event(sev="alert"))
        assert len(tl.by_severity("info")) == 1
        assert len(tl.warnings_and_alerts()) == 2

    def test_to_text_empty(self):
        tl = EventTimeline()
        assert "No events" in tl.to_text()

    def test_reset(self):
        tl = EventTimeline()
        tl.add(self._event())
        tl.reset()
        assert tl.count == 0

    def test_recent(self):
        tl = EventTimeline()
        for i in range(30):
            tl.add(self._event(f"e{i}"))
        assert len(tl.recent(5)) == 5

    def test_get_summary_structure(self):
        tl = EventTimeline()
        tl.add(self._event(sev="warning"))
        s = tl.get_summary()
        assert s["total"] == 1
        assert "types" in s
        assert "severities" in s

    def test_to_list(self, sample_event):
        tl = EventTimeline()
        tl.add(sample_event)
        lst = tl.to_list()
        assert len(lst) == 1
        assert lst[0]["event_type"] == "object_appeared"
