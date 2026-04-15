"""Tests for orchestration layer — node functions, routing, and decision logic."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.orchestration import (
    _EMPTY_STATE,
    _dets_from_state,
    _frame_result_from_state,
    _scene_state,
    node_ingest,
    node_run_cv,
    node_extract_events,
    node_update_memory,
    node_detect_change,
    node_create_alert,
    node_decide_reasoning,
    node_decide_llm,        # backward-compat alias
    node_call_llm,
    node_finalize,
    node_qa_cv,
    _node_qa_cv,             # backward-compat alias
    node_gather_report_context,
    route_after_ingest,
    route_after_change,
    route_after_reasoning,
    route_qa_after_cv,
    reset_session,
    empty_state,
    PipelineState,
)
import src.orchestration.nodes as _nodes


class TestState:
    def test_empty_state_has_all_defaults(self):
        s = empty_state()
        assert s["mode"] == "image"
        assert s["current_frame"] is None
        assert s["detections"] == []
        assert s["has_notable_change"] is False
        assert s["change_severity"] == "none"
        assert s["_node_trace"] == []

    def test_backward_compat_empty_state(self):
        assert _EMPTY_STATE["mode"] == "image"
        assert _EMPTY_STATE["cv_ran"] is False


class TestHelpers:
    def test_dets_from_state_empty(self):
        assert _dets_from_state({"detections": []}) == []

    def test_dets_from_state_roundtrip(self, sample_detection):
        state = {"detections": [sample_detection.to_dict()]}
        dets = _dets_from_state(state)
        assert len(dets) == 1
        assert dets[0].class_name == "person"

    def test_frame_result_from_state(self):
        state = {**empty_state(), "detections": [{"class_id": 0, "class_name": "a", "confidence": 0.9, "bbox": [0, 0, 1, 1]}],
                 "object_counts": {"a": 1}, "frame_index": 5}
        fr = _frame_result_from_state(state)
        assert fr.frame_index == 5
        assert len(fr.detections) == 1


class TestNodeIngest:
    def test_valid_frame(self, sample_frame):
        state = {**empty_state(), "current_frame": sample_frame, "mode": "live"}
        out = node_ingest(state)
        assert out.get("error") == ""
        assert "ingest" in out["_node_trace"]

    def test_missing_frame(self):
        state = {**empty_state(), "current_frame": None}
        out = node_ingest(state)
        assert "No frame" in out["error"]
        assert "ingest" in out["_node_trace"]


class TestNodeRunCV:
    def test_no_frame_returns_error(self):
        state = {**empty_state(), "current_frame": None}
        out = node_run_cv(state)
        assert out["error"]

    def test_no_detector_returns_error(self, sample_frame):
        reset_session()
        state = {**empty_state(), "current_frame": sample_frame}
        out = node_run_cv(state)
        assert "Detector" in out["error"]

    def test_trace_added(self, sample_frame):
        reset_session()
        state = {**empty_state(), "current_frame": sample_frame}
        out = node_run_cv(state)
        assert "run_cv" in out["_node_trace"]


class TestNodeExtractEvents:
    def test_empty_detections(self):
        state = {**empty_state(), "detections": [], "frame_index": 0}
        out = node_extract_events(state)
        assert "new_events" in out
        assert "total_event_count" in out

    def test_trace(self):
        state = {**empty_state(), "detections": [], "frame_index": 0}
        out = node_extract_events(state)
        assert "extract_events" in out["_node_trace"]


class TestNodeUpdateMemory:
    def test_updates_scene(self):
        state = {**empty_state(), "detections": [], "frame_index": 0}
        out = node_update_memory(state)
        assert "scene_description" in out

    def test_trace(self):
        state = {**empty_state(), "detections": [], "frame_index": 0}
        out = node_update_memory(state)
        assert "update_memory" in out["_node_trace"]


class TestNodeDetectChange:
    def test_no_events_means_no_change(self):
        state = {**empty_state(), "new_events": []}
        out = node_detect_change(state)
        assert out["has_notable_change"] is False
        assert out["change_severity"] == "none"

    def test_info_event_is_info_severity(self):
        state = {**empty_state(), "new_events": [{"severity": "info", "event_type": "appeared"}]}
        out = node_detect_change(state)
        assert out["has_notable_change"] is True
        assert out["change_severity"] == "info"

    def test_warning_event_is_warning_severity(self):
        state = {**empty_state(), "new_events": [
            {"severity": "info", "event_type": "appeared"},
            {"severity": "warning", "event_type": "crowding"},
        ]}
        out = node_detect_change(state)
        assert out["change_severity"] == "warning"

    def test_alert_severity_takes_precedence(self):
        state = {**empty_state(), "new_events": [
            {"severity": "warning", "event_type": "crowding"},
            {"severity": "alert", "event_type": "intrusion"},
        ]}
        out = node_detect_change(state)
        assert out["change_severity"] == "alert"


class TestNodeDecideReasoning:
    def test_question_triggers_llm(self):
        state = {**empty_state(), "user_question": "What do you see?"}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True

    def test_no_trigger(self):
        state = {**empty_state()}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is False

    def test_warning_severity_triggers_anomaly(self):
        state = {**empty_state(), "change_severity": "warning"}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "anomaly"

    def test_alert_severity_triggers_anomaly(self):
        state = {**empty_state(), "change_severity": "alert"}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "anomaly"

    def test_threshold_crossing(self):
        """LLM should only trigger when crossing a new threshold, not every frame."""
        reset_session()
        _nodes._last_llm_summary_bucket = 0
        state = {**empty_state(), "total_event_count": 3, "new_events": []}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        # Same count again → should NOT trigger
        out2 = node_decide_reasoning(state)
        assert out2["llm_needed"] is False

    def test_threshold_bucket_handles_jumps(self):
        reset_session()
        _nodes._last_llm_summary_bucket = 0
        out = node_decide_reasoning({**empty_state(), "total_event_count": 5, "new_events": []})
        assert out["llm_needed"] is True
        out2 = node_decide_reasoning({**empty_state(), "total_event_count": 6, "new_events": []})
        assert out2["llm_needed"] is True

    def test_live_mode_suppresses_periodic_summary(self):
        """In live mode, periodic summaries are suppressed to keep latency low."""
        reset_session()
        _nodes._last_llm_summary_bucket = 0
        state = {**empty_state(), "mode": "live", "total_event_count": 10, "new_events": []}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is False

    def test_live_mode_still_triggers_on_warning(self):
        """Warnings should still trigger LLM even in live mode."""
        state = {**empty_state(), "mode": "live", "change_severity": "warning"}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "anomaly"

    def test_live_mode_still_triggers_on_explicit_question(self):
        state = {**empty_state(), "mode": "live", "user_question": "What is happening?"}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True

    def test_image_mode_allows_periodic_summary(self):
        """Image/video modes should still allow periodic LLM summaries."""
        reset_session()
        _nodes._last_llm_summary_bucket = 0
        state = {**empty_state(), "mode": "image", "total_event_count": 3, "new_events": []}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "summarize"

    def test_backward_compat_alias(self):
        """node_decide_llm should be the same as node_decide_reasoning."""
        assert node_decide_llm is node_decide_reasoning


class TestNodeCallLLM:
    def test_no_reasoner(self):
        reset_session()
        state = {**empty_state(), "reasoning_task": "describe"}
        out = node_call_llm(state)
        assert "No LLM configured" in out["llm_response"]


class TestNodeCreateAlert:
    def test_warning_creates_alert(self):
        state = {**empty_state(), "new_events": [
            {"event_type": "crowding", "severity": "warning", "description": "many"}
        ], "alerts": []}
        out = node_create_alert(state)
        assert len(out["alerts"]) == 1

    def test_info_ignored(self):
        state = {**empty_state(), "new_events": [
            {"event_type": "appeared", "severity": "info", "description": "ok"}
        ], "alerts": []}
        out = node_create_alert(state)
        assert len(out["alerts"]) == 0


class TestNodeFinalize:
    def test_returns_trace(self):
        state = {**empty_state()}
        out = node_finalize(state)
        assert "finalize" in out["_node_trace"]


class TestNodeGatherReportContext:
    def test_gathers_scene_and_timeline(self):
        reset_session()
        out = node_gather_report_context({**empty_state()})
        assert "scene_description" in out
        assert "event_history_text" in out
        assert "object_counts" in out
        assert "gather_report_context" in out["_node_trace"]


class TestRouting:
    def test_route_after_ingest_error(self):
        assert route_after_ingest({"error": "No frame"}) == "finalize"

    def test_route_after_ingest_ok(self):
        assert route_after_ingest({"error": ""}) == "run_cv"

    def test_route_after_change_warning(self):
        assert route_after_change({"change_severity": "warning"}) == "create_alert"

    def test_route_after_change_alert(self):
        assert route_after_change({"change_severity": "alert"}) == "create_alert"

    def test_route_after_change_info(self):
        assert route_after_change({"change_severity": "info"}) == "decide_reasoning"

    def test_route_after_change_none(self):
        assert route_after_change({"change_severity": "none"}) == "decide_reasoning"

    def test_route_after_reasoning_needed(self):
        assert route_after_reasoning({"llm_needed": True}) == "call_llm"

    def test_route_after_reasoning_skip(self):
        assert route_after_reasoning({"llm_needed": False}) == "finalize"

    def test_route_qa_after_cv_ran(self):
        assert route_qa_after_cv({"cv_ran": True}) == "update_memory"

    def test_route_qa_after_cv_skipped(self):
        assert route_qa_after_cv({"cv_ran": False}) == "call_llm"


class TestQACV:
    def test_no_frame_passes_through(self):
        reset_session()
        state = {**empty_state(), "current_frame": None}
        out = node_qa_cv(state)
        assert out.get("error", "") == ""
        assert out["cv_ran"] is False

    def test_with_frame_but_no_detector(self, sample_frame):
        reset_session()
        state = {**empty_state(), "current_frame": sample_frame}
        out = node_qa_cv(state)
        assert out.get("error", "") == ""
        assert out["cv_ran"] is False

    def test_backward_compat_alias(self):
        assert _node_qa_cv is node_qa_cv

    def test_route_skips_memory_when_cv_not_run(self):
        assert route_qa_after_cv({**empty_state(), "cv_ran": False}) == "call_llm"

    def test_no_frame_qa_does_not_mutate_scene_memory(self):
        reset_session()
        before = _scene_state.total_frames
        out = node_qa_cv({**empty_state(), "current_frame": None})
        assert route_qa_after_cv(out) == "call_llm"
        assert _scene_state.total_frames == before
