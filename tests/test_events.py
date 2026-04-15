"""Tests for the events module — EventExtractor."""

import pytest

from src.vision.detector import Detection, FrameResult
from src.vision.events import EventExtractor, SceneEvent


class TestEventExtractor:
    def _frame(self, counts, dets=None, idx=0):
        dets = dets or []
        return FrameResult(frame_index=idx, detections=dets, object_counts=counts)

    def test_new_class_emits_appeared(self):
        ex = EventExtractor(cooldown_seconds=0)
        events = ex.extract(self._frame({"person": 1}))
        types = [e.event_type for e in events]
        assert "object_appeared" in types

    def test_class_left_emits_event(self):
        ex = EventExtractor(cooldown_seconds=0)
        ex.extract(self._frame({"car": 2}))  # establish prev
        events = ex.extract(self._frame({}))  # car left
        types = [e.event_type for e in events]
        assert "object_left" in types

    def test_count_change(self):
        ex = EventExtractor(cooldown_seconds=0)
        ex.extract(self._frame({"person": 5}))
        events = ex.extract(self._frame({"person": 10}))
        types = [e.event_type for e in events]
        assert "count_change" in types

    def test_crowding(self):
        ex = EventExtractor(cooldown_seconds=0)
        events = ex.extract(self._frame({"person": 15}))
        types = [e.event_type for e in events]
        assert "crowding" in types

    def test_cooldown_suppresses(self):
        ex = EventExtractor(cooldown_seconds=9999)
        events1 = ex.extract(self._frame({"x": 1}))
        ex.extract(self._frame({}))  # x left
        events2 = ex.extract(self._frame({"x": 1}))
        # cooldown prevents second emission
        appeared_count = sum(1 for e in events2 if e.event_type == "object_appeared")
        assert appeared_count == 0

    def test_reset(self):
        ex = EventExtractor(cooldown_seconds=0)
        ex.extract(self._frame({"a": 1}))
        ex.reset()
        events = ex.extract(self._frame({"a": 1}))
        assert len(events) > 0  # "appeared" again after reset


class TestSceneEvent:
    def test_to_dict(self):
        e = SceneEvent(event_type="test", description="desc", severity="info")
        d = e.to_dict()
        assert d["event_type"] == "test"
        assert "timestamp" in d
