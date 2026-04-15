"""Tests for orchestration layer — node functions and decision logic."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.orchestration import (
    _EMPTY_STATE,
    _dets_from_state,
    _frame_result_from_state,
    _scene_state,
    node_run_cv,
    node_extract_events,
    node_update_memory,
    node_decide_llm,
    node_call_llm,
    node_create_alert,
    _node_qa_cv,
    route_qa_after_cv,
    reset_session,
)


class TestHelpers:
    def test_dets_from_state_empty(self):
        assert _dets_from_state({"detections": []}) == []

    def test_dets_from_state_roundtrip(self, sample_detection):
        state = {"detections": [sample_detection.to_dict()]}
        dets = _dets_from_state(state)
        assert len(dets) == 1
        assert dets[0].class_name == "person"

    def test_frame_result_from_state(self):
        state = {**_EMPTY_STATE, "detections": [{"class_id": 0, "class_name": "a", "confidence": 0.9, "bbox": [0, 0, 1, 1]}],
                 "object_counts": {"a": 1}, "frame_index": 5}
        fr = _frame_result_from_state(state)
        assert fr.frame_index == 5
        assert len(fr.detections) == 1


class TestNodeRunCV:
    def test_no_frame_returns_error(self):
        state = {**_EMPTY_STATE, "current_frame": None}
        out = node_run_cv(state)
        assert out["error"]

    def test_no_detector_returns_error(self, sample_frame):
        reset_session()
        state = {**_EMPTY_STATE, "current_frame": sample_frame}
        out = node_run_cv(state)
        assert "Detector" in out["error"]


class TestNodeExtractEvents:
    def test_empty_detections(self):
        state = {**_EMPTY_STATE, "detections": [], "frame_index": 0}
        out = node_extract_events(state)
        assert "new_events" in out


class TestNodeUpdateMemory:
    def test_updates_scene(self):
        state = {**_EMPTY_STATE, "detections": [], "frame_index": 0}
        out = node_update_memory(state)
        assert "scene_description" in out


class TestNodeDecideLLM:
    def test_question_triggers_llm(self):
        state = {**_EMPTY_STATE, "user_question": "What do you see?"}
        out = node_decide_llm(state)
        assert out["llm_needed"] is True

    def test_no_trigger(self):
        state = {**_EMPTY_STATE}
        out = node_decide_llm(state)
        assert out["llm_needed"] is False

    def test_warning_triggers_anomaly(self):
        state = {**_EMPTY_STATE, "new_events": [{"severity": "warning", "event_type": "crowding"}]}
        out = node_decide_llm(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "anomaly"

    def test_threshold_crossing(self):
        """LLM should only trigger when crossing a new threshold, not every frame."""
        reset_session()
        import src.orchestration as orch
        orch._last_llm_summary_bucket = 0
        # At count=3 with threshold=3 → should trigger
        state = {**_EMPTY_STATE, "total_event_count": 3, "new_events": []}
        out = node_decide_llm(state)
        assert out["llm_needed"] is True
        # Same count again → should NOT trigger
        out2 = node_decide_llm(state)
        assert out2["llm_needed"] is False

    def test_threshold_bucket_handles_jumps(self):
        reset_session()
        import src.orchestration as orch
        orch._last_llm_summary_bucket = 0
        out = node_decide_llm({**_EMPTY_STATE, "total_event_count": 5, "new_events": []})
        assert out["llm_needed"] is True
        out2 = node_decide_llm({**_EMPTY_STATE, "total_event_count": 6, "new_events": []})
        assert out2["llm_needed"] is True

    def test_live_mode_suppresses_periodic_summary(self):
        """In live mode, periodic summaries are suppressed to keep latency low."""
        reset_session()
        import src.orchestration as orch
        orch._last_llm_summary_bucket = 0
        state = {**_EMPTY_STATE, "mode": "live", "total_event_count": 10, "new_events": []}
        out = node_decide_llm(state)
        assert out["llm_needed"] is False

    def test_live_mode_still_triggers_on_warning(self):
        """Warnings should still trigger LLM even in live mode."""
        state = {**_EMPTY_STATE, "mode": "live",
                 "new_events": [{"severity": "warning", "event_type": "crowding"}]}
        out = node_decide_llm(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "anomaly"

    def test_live_mode_still_triggers_on_explicit_question(self):
        state = {**_EMPTY_STATE, "mode": "live", "user_question": "What is happening?"}
        out = node_decide_llm(state)
        assert out["llm_needed"] is True

    def test_image_mode_allows_periodic_summary(self):
        """Image/video modes should still allow periodic LLM summaries."""
        reset_session()
        import src.orchestration as orch
        orch._last_llm_summary_bucket = 0
        state = {**_EMPTY_STATE, "mode": "image", "total_event_count": 3, "new_events": []}
        out = node_decide_llm(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "summarize"


class TestNodeCallLLM:
    def test_no_reasoner(self):
        reset_session()
        state = {**_EMPTY_STATE, "reasoning_task": "describe"}
        out = node_call_llm(state)
        assert "No LLM configured" in out["llm_response"]


class TestNodeCreateAlert:
    def test_warning_creates_alert(self):
        state = {**_EMPTY_STATE, "new_events": [
            {"event_type": "crowding", "severity": "warning", "description": "many"}
        ], "alerts": []}
        out = node_create_alert(state)
        assert len(out["alerts"]) == 1

    def test_info_ignored(self):
        state = {**_EMPTY_STATE, "new_events": [
            {"event_type": "appeared", "severity": "info", "description": "ok"}
        ], "alerts": []}
        out = node_create_alert(state)
        assert len(out["alerts"]) == 0


class TestQACV:
    def test_no_frame_passes_through(self):
        reset_session()
        state = {**_EMPTY_STATE, "current_frame": None}
        out = _node_qa_cv(state)
        # Should pass through without error
        assert out.get("error", "") == ""
        assert out["cv_ran"] is False

    def test_with_frame_but_no_detector(self, sample_frame):
        reset_session()
        state = {**_EMPTY_STATE, "current_frame": sample_frame}
        out = _node_qa_cv(state)
        # No detector → pass through silently
        assert out.get("error", "") == ""
        assert out["cv_ran"] is False

    def test_route_skips_memory_when_cv_not_run(self):
        assert route_qa_after_cv({**_EMPTY_STATE, "cv_ran": False}) == "call_llm"

    def test_no_frame_qa_does_not_mutate_scene_memory(self):
        reset_session()
        before = _scene_state.total_frames
        out = _node_qa_cv({**_EMPTY_STATE, "current_frame": None})
        assert route_qa_after_cv(out) == "call_llm"
        assert _scene_state.total_frames == before
