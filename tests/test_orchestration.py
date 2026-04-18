"""Tests for orchestration layer — node functions, routing, and decision logic."""

from src.orchestration import (
    _dets_from_state,
    _frame_result_from_state,
    empty_state,
    node_call_llm,
    node_create_alert,
    node_decide_reasoning,
    node_detect_change,
    node_extract_events,
    node_finalize,
    node_gather_report_context,
    node_ingest,
    node_qa_cv,
    node_run_cv,
    node_update_memory,
    reset_session,
    route_after_change,
    route_after_ingest,
    route_after_reasoning,
    route_qa_after_cv,
)
from src.orchestration.graph import route_after_cv
from src.orchestration.nodes import get_scene_state


class TestState:
    def test_empty_state_has_all_defaults(self):
        s = empty_state()
        assert s["mode"] == "image"
        assert s["current_frame"] is None
        assert s["detections"] == []
        assert s["has_notable_change"] is False
        assert s["change_severity"] == "none"
        assert s["_node_trace"] == []
        assert s["last_llm_summary_bucket"] == 0


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

    def test_threshold_crossing_emits_bucket_in_state(self):
        """Node must emit the new bucket value in its partial-state return."""
        state = {**empty_state(), "total_event_count": 3, "last_llm_summary_bucket": 0}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "summarize"
        assert out["last_llm_summary_bucket"] == 1

    def test_threshold_same_bucket_does_not_retrigger(self):
        """Same bucket as last time → no trigger."""
        state = {**empty_state(), "total_event_count": 3, "last_llm_summary_bucket": 1}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is False

    def test_threshold_bucket_handles_jumps(self):
        """Crossing multiple buckets at once still fires once, with the new bucket."""
        out = node_decide_reasoning({
            **empty_state(), "total_event_count": 9, "last_llm_summary_bucket": 0,
        })
        assert out["llm_needed"] is True
        assert out["last_llm_summary_bucket"] == 3

    def test_live_mode_suppresses_periodic_summary(self):
        state = {**empty_state(), "mode": "live", "total_event_count": 10}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is False

    def test_live_mode_still_triggers_on_warning(self):
        state = {**empty_state(), "mode": "live", "change_severity": "warning"}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "anomaly"

    def test_live_mode_still_triggers_on_explicit_question(self):
        state = {**empty_state(), "mode": "live", "user_question": "What is happening?"}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True

    def test_image_mode_allows_periodic_summary(self):
        state = {**empty_state(), "mode": "image", "total_event_count": 3, "last_llm_summary_bucket": 0}
        out = node_decide_reasoning(state)
        assert out["llm_needed"] is True
        assert out["reasoning_task"] == "summarize"


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

    def test_route_after_cv_error_short_circuits(self):
        """A CV error must skip extract_events/update_memory to avoid polluting scene state."""
        assert route_after_cv({"error": "Detector not initialised"}) == "finalize"

    def test_route_after_cv_ok(self):
        assert route_after_cv({"error": ""}) == "extract_events"

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

    def test_route_qa_after_cv_always_call_llm(self):
        """Q&A graph no longer runs update_memory — it always goes straight to call_llm."""
        assert route_qa_after_cv({"cv_ran": True}) == "call_llm"
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

    def test_no_frame_qa_does_not_mutate_scene_memory(self):
        reset_session()
        before = get_scene_state().total_frames
        node_qa_cv({**empty_state(), "current_frame": None})
        assert get_scene_state().total_frames == before
