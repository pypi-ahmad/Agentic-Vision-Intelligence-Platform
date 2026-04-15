"""LangGraph orchestration — state definition, node functions, and compiled workflows."""

from __future__ import annotations

import json
import logging
from typing import Any

from langgraph.graph import END, StateGraph

from config import get_settings, MODE_USES_TRACKING
from src.vision.detector import Detection, FrameResult, VisionDetector
from src.vision.events import EventExtractor
from src.memory.scene_state import SceneState
from src.memory.event_timeline import EventTimeline
from src.providers.base import LLMProvider
from src.reasoning.reasoner import Reasoner

logger = logging.getLogger(__name__)

# ======================================================================
# Shared singletons (stateful across invocations within a session).
# They are reset via reset_session().
# ======================================================================

_cfg = get_settings()
_event_extractor = EventExtractor(cooldown_seconds=_cfg.event_cooldown_seconds)
_scene_state = SceneState(window_seconds=_cfg.memory_window_seconds)
_event_timeline = EventTimeline()
_last_llm_summary_bucket: int = 0  # track summary thresholds to avoid LLM spam

# Detector and reasoner are set dynamically from the UI
_detector: VisionDetector | None = None
_reasoner: Reasoner | None = None


def set_detector(variant: str, confidence: float | None = None) -> None:
    global _detector
    _detector = VisionDetector(variant=variant, confidence=confidence)


def set_reasoner(provider: LLMProvider, model: str) -> None:
    global _reasoner
    _reasoner = Reasoner(provider, model)


def clear_reasoner() -> None:
    global _reasoner
    _reasoner = None


def get_scene_state() -> SceneState:
    return _scene_state


def get_event_timeline() -> EventTimeline:
    return _event_timeline


def get_reasoner() -> Reasoner | None:
    return _reasoner


def reset_session() -> None:
    global _detector, _reasoner, _last_llm_summary_bucket
    _event_extractor.reset()
    _scene_state.reset()
    _event_timeline.reset()
    _detector = None
    _reasoner = None
    _last_llm_summary_bucket = 0


# ======================================================================
# NODE FUNCTIONS — each takes and returns a dict (graph state)
# ======================================================================

def _dets_from_state(state: dict[str, Any]) -> list[Detection]:
    """Reconstruct Detection objects from serialised dicts in graph state."""
    return [
        Detection(
            d.get("class_id", 0), d.get("class_name", "?"), d.get("confidence", 0),
            tuple(d.get("bbox", [0, 0, 0, 0])), d.get("track_id"),
        )
        for d in state.get("detections", [])
    ]


def _frame_result_from_state(state: dict[str, Any]) -> FrameResult:
    return FrameResult(
        frame_index=state.get("frame_index", 0),
        detections=_dets_from_state(state),
        object_counts=state.get("object_counts", {}),
    )

def node_run_cv(state: dict[str, Any]) -> dict[str, Any]:
    """Run YOLO26 detection/tracking on the current frame."""
    frame = state.get("current_frame")
    if frame is None:
        return {**state, "error": "No frame for CV"}
    if _detector is None:
        return {**state, "error": "Detector not initialised — select a CV model first"}
    mode = state.get("mode", "image")
    try:
        if MODE_USES_TRACKING.get(mode, False):
            result = _detector.track(frame, persist=True)
        else:
            result = _detector.detect(frame)
    except Exception as exc:
        return {**state, "error": f"CV error: {exc}"}
    return {
        **state,
        "detections": [d.to_dict() for d in result.detections],
        "object_counts": result.object_counts,
        "detection_summary": result.summary_line,
        "annotated_frame": result.annotated_frame,
        "error": "",
    }


def node_extract_events(state: dict[str, Any]) -> dict[str, Any]:
    fr = _frame_result_from_state(state)
    events = _event_extractor.extract(fr, frame_index=state.get("frame_index", 0))
    _event_timeline.add_many(events)
    return {**state, "new_events": [e.to_dict() for e in events],
            "event_history_text": _event_timeline.to_text(), "total_event_count": _event_timeline.count}


def node_update_memory(state: dict[str, Any]) -> dict[str, Any]:
    fr = _frame_result_from_state(state)
    _scene_state.update(fr)
    return {**state, "scene_description": _scene_state.get_description(),
            "scene_summary": _scene_state.get_summary()}


def node_decide_llm(state: dict[str, Any]) -> dict[str, Any]:
    global _last_llm_summary_bucket
    mode = state.get("mode", "image")
    # Explicit task or question → always call LLM
    if state.get("reasoning_task") or state.get("user_question"):
        return {**state, "llm_needed": True}
    # Warning/alert events → auto-trigger anomaly reasoning
    warnings = [e for e in state.get("new_events", []) if e.get("severity") in ("warning", "alert")]
    if warnings:
        return {**state, "llm_needed": True, "reasoning_task": "anomaly"}
    # Live mode: suppress periodic summaries to keep frame processing fast.
    # Users can still get LLM analysis on-demand via Q&A or Report.
    if mode == "live":
        return {**state, "llm_needed": False}
    # Periodic summary: only trigger when crossing a new threshold bucket.
    total = state.get("total_event_count", 0)
    threshold = _cfg.llm_trigger_threshold
    summary_bucket = total // threshold if threshold > 0 else 0
    if total > 0 and summary_bucket > _last_llm_summary_bucket:
        _last_llm_summary_bucket = summary_bucket
        return {**state, "llm_needed": True, "reasoning_task": "summarize"}
    return {**state, "llm_needed": False}


def node_call_llm(state: dict[str, Any]) -> dict[str, Any]:
    if _reasoner is None:
        return {**state, "llm_response": "(No LLM configured)", "error": "Select a provider and model first"}
    task = state.get("reasoning_task", "describe")
    frame = state.get("current_frame")
    scene = state.get("scene_description", "")
    events_txt = state.get("event_history_text", "")
    det_sum = state.get("detection_summary", "")
    q = state.get("user_question", "")
    oc = json.dumps(state.get("object_counts", {}))
    try:
        if task == "qa" and q:
            resp = _reasoner.answer_question(q, image=frame, scene_description=scene,
                                              recent_events=events_txt, detection_data=det_sum)
        elif task == "summarize":
            resp = _reasoner.summarize_events(events_text=events_txt, scene_state=scene)
        elif task == "anomaly":
            resp = _reasoner.reason_anomalies(detection_data=det_sum, events_text=events_txt, scene_description=scene)
        elif task == "report":
            resp = _reasoner.generate_report(session_id=state.get("source_id", "session"),
                                              duration=state.get("duration", "active"), total_frames=state.get("scene_summary", {}).get("total_frames", 0),
                                              scene_summary=scene, events_text=events_txt, object_stats=oc)
        elif task == "alert":
            alerts = state.get("alerts", [])
            if alerts:
                a = alerts[-1]
                resp = _reasoner.explain_alert(alert_type=a.get("event_type", "?"), severity=a.get("severity", "info"),
                                                description=a.get("description", ""), scene_context=scene, recent_events=events_txt)
            else:
                resp = "No alerts to explain."
        else:
            resp = _reasoner.describe_scene(image=frame, detection_summary=det_sum, object_counts=oc, tracked_objects=scene)
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        return {**state, "llm_response": "", "error": f"LLM error: {exc}"}
    result = {**state, "llm_response": resp}
    if task == "qa":
        result["answer"] = resp
    if task == "report":
        result["report"] = resp
    return result


def node_create_alert(state: dict[str, Any]) -> dict[str, Any]:
    alerts = list(state.get("alerts", []))
    for ev in state.get("new_events", []):
        if ev.get("severity") in ("warning", "alert"):
            alerts.append(ev)
    return {**state, "alerts": alerts}


# ---- routing ---------------------------------------------------------

def route_llm(state: dict[str, Any]) -> str:
    return "call_llm" if state.get("llm_needed") else "skip"


def route_qa_after_cv(state: dict[str, Any]) -> str:
    return "update_memory" if state.get("cv_ran") else "call_llm"


# ======================================================================
# GRAPH DEFINITIONS
# ======================================================================

def build_perception_graph() -> StateGraph:
    g = StateGraph(dict)
    g.add_node("run_cv", node_run_cv)
    g.add_node("extract_events", node_extract_events)
    g.add_node("update_memory", node_update_memory)
    g.add_node("create_alert", node_create_alert)
    g.add_node("decide_llm", node_decide_llm)
    g.add_node("call_llm", node_call_llm)
    g.set_entry_point("run_cv")
    g.add_edge("run_cv", "extract_events")
    g.add_edge("extract_events", "update_memory")
    g.add_edge("update_memory", "create_alert")
    g.add_edge("create_alert", "decide_llm")
    g.add_conditional_edges("decide_llm", route_llm, {"call_llm": "call_llm", "skip": END})
    g.add_edge("call_llm", END)
    return g


def build_qa_graph() -> StateGraph:
    """QA graph: optionally runs CV if a frame is provided, then calls LLM."""
    g = StateGraph(dict)
    g.add_node("maybe_cv", _node_qa_cv)
    g.add_node("update_memory", node_update_memory)
    g.add_node("call_llm", node_call_llm)
    g.set_entry_point("maybe_cv")
    g.add_conditional_edges("maybe_cv", route_qa_after_cv, {"update_memory": "update_memory", "call_llm": "call_llm"})
    g.add_edge("update_memory", "call_llm")
    g.add_edge("call_llm", END)
    return g


def _node_qa_cv(state: dict[str, Any]) -> dict[str, Any]:
    """Run CV only if a frame is present; otherwise return the state unchanged."""
    if state.get("current_frame") is not None and _detector is not None:
        return {**node_run_cv(state), "cv_ran": True}
    return {**state, "cv_ran": False}


def build_report_graph() -> StateGraph:
    g = StateGraph(dict)
    g.add_node("call_llm", node_call_llm)
    g.set_entry_point("call_llm")
    g.add_edge("call_llm", END)
    return g


# ======================================================================
# COMPILED CACHES
# ======================================================================

_compiled: dict[str, Any] = {}


def _get(name: str, builder):
    if name not in _compiled:
        _compiled[name] = builder().compile()
    return _compiled[name]


def get_perception_wf():
    return _get("perception", build_perception_graph)


def get_qa_wf():
    return _get("qa", build_qa_graph)


def get_report_wf():
    return _get("report", build_report_graph)


# ======================================================================
# HIGH-LEVEL INVOCATION
# ======================================================================

_EMPTY_STATE: dict[str, Any] = dict(
    mode="image", current_frame=None, source_id="", frame_index=0,
    detections=[], object_counts={}, detection_summary="", annotated_frame=None,
    new_events=[], event_history_text="", total_event_count=0,
    scene_description="", scene_summary={},
    llm_needed=False, llm_response="", reasoning_task="",
    user_question="", answer="",
    alerts=[], report="",
    cv_ran=False,
    error="",
)


def process_frame(frame, *, mode: str = "image", source_id: str = "", frame_index: int = 0) -> dict[str, Any]:
    s = {**_EMPTY_STATE, "mode": mode, "current_frame": frame, "source_id": source_id, "frame_index": frame_index}
    return get_perception_wf().invoke(s)


def ask_question(question: str, *, frame=None, mode: str = "image") -> dict[str, Any]:
    s = {**_EMPTY_STATE, "mode": mode, "current_frame": frame,
         "llm_needed": True, "reasoning_task": "qa", "user_question": question,
         "event_history_text": _event_timeline.to_text(), "total_event_count": _event_timeline.count,
         "scene_description": _scene_state.get_description(), "scene_summary": _scene_state.get_summary()}
    return get_qa_wf().invoke(s)


def generate_report(*, session_id: str = "session", duration: str = "active") -> dict[str, Any]:
    s = {**_EMPTY_STATE, "llm_needed": True, "reasoning_task": "report", "source_id": session_id,
         "object_counts": _scene_state.current_counts,
         "event_history_text": _event_timeline.to_text(), "total_event_count": _event_timeline.count,
        "duration": duration,
         "scene_description": _scene_state.get_description(), "scene_summary": _scene_state.get_summary()}
    return get_report_wf().invoke(s)
