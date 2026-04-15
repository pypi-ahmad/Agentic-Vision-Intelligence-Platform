"""Graph node functions — each takes the full PipelineState and returns only
the fields it modifies.  LangGraph merges the returned dict into the running
state via last-writer-wins (or append for ``_node_trace``).

Singletons (detector, reasoner, event extractor, scene memory, event timeline)
are managed here as session-scoped module globals.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import get_settings, MODE_USES_TRACKING
from src.vision.detector import Detection, FrameResult, VisionDetector
from src.vision.events import EventExtractor
from src.memory.scene_state import SceneState
from src.memory.event_timeline import EventTimeline
from src.providers.base import LLMProvider
from src.reasoning.reasoner import Reasoner

logger = logging.getLogger(__name__)

_cfg = get_settings()

# ======================================================================
# Session-scoped singletons (reset via reset_all)
# ======================================================================

_event_extractor = EventExtractor(cooldown_seconds=_cfg.event_cooldown_seconds)
_scene_state = SceneState(window_seconds=_cfg.memory_window_seconds)
_event_timeline = EventTimeline()
_last_llm_summary_bucket: int = 0

_detector: VisionDetector | None = None
_reasoner: Reasoner | None = None


# ---- Singleton accessors / mutators ----------------------------------

def set_detector(variant: str, confidence: float | None = None) -> None:
    global _detector
    _detector = VisionDetector(variant=variant, confidence=confidence)


def set_reasoner_obj(provider: LLMProvider, model: str) -> None:
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


def reset_all() -> None:
    global _detector, _reasoner, _last_llm_summary_bucket
    _event_extractor.reset()
    _scene_state.reset()
    _event_timeline.reset()
    _detector = None
    _reasoner = None
    _last_llm_summary_bucket = 0


# ======================================================================
# Helpers
# ======================================================================

def _trace(name: str) -> dict[str, Any]:
    """Return a partial-state dict that appends *name* to _node_trace."""
    return {"_node_trace": [name]}


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


# ======================================================================
# Perception nodes
# ======================================================================

def node_ingest(state: dict[str, Any]) -> dict[str, Any]:
    """Validate and log frame ingestion.

    Sets ``error`` if the frame is missing so downstream routing can
    short-circuit to ``finalize``.
    """
    frame = state.get("current_frame")
    mode = state.get("mode", "image")
    fi = state.get("frame_index", 0)

    if frame is None:
        logger.debug("[ingest] no frame (mode=%s) → will short-circuit", mode)
        return {**_trace("ingest"), "error": "No frame for CV"}

    h, w = frame.shape[:2] if hasattr(frame, "shape") else (0, 0)
    logger.debug("[ingest] mode=%s frame=%dx%d idx=%d", mode, w, h, fi)
    return {**_trace("ingest"), "error": ""}


def node_run_cv(state: dict[str, Any]) -> dict[str, Any]:
    """Run YOLO26 detection/tracking on the current frame.

    Uses ``MODE_USES_TRACKING`` to pick ``track()`` vs ``detect()``
    based on the active mode.
    """
    frame = state.get("current_frame")
    if frame is None:
        return {**_trace("run_cv"), "error": "No frame for CV"}
    if _detector is None:
        return {**_trace("run_cv"), "error": "Detector not initialised — select a CV model first"}

    mode = state.get("mode", "image")
    try:
        if MODE_USES_TRACKING.get(mode, False):
            result = _detector.track(frame, persist=True)
        else:
            result = _detector.detect(frame)
    except Exception as exc:
        logger.error("[run_cv] CV inference failed: %s", exc)
        return {**_trace("run_cv"), "error": f"CV error: {exc}"}

    n = len(result.detections)
    logger.debug("[run_cv] %d detections, counts=%s", n, result.object_counts)
    return {
        **_trace("run_cv"),
        "detections": [d.to_dict() for d in result.detections],
        "object_counts": result.object_counts,
        "detection_summary": result.summary_line,
        "annotated_frame": result.annotated_frame,
        "error": "",
    }


def node_extract_events(state: dict[str, Any]) -> dict[str, Any]:
    """Extract events by comparing current detections to previous state."""
    fr = _frame_result_from_state(state)
    events = _event_extractor.extract(fr, frame_index=state.get("frame_index", 0))
    _event_timeline.add_many(events)

    logger.debug(
        "[extract_events] %d new events, timeline_total=%d",
        len(events), _event_timeline.count,
    )
    return {
        **_trace("extract_events"),
        "new_events": [e.to_dict() for e in events],
        "event_history_text": _event_timeline.to_text(),
        "total_event_count": _event_timeline.count,
    }


def node_update_memory(state: dict[str, Any]) -> dict[str, Any]:
    """Update the rolling scene memory with current detections."""
    fr = _frame_result_from_state(state)
    _scene_state.update(fr)

    logger.debug(
        "[update_memory] tracked=%d, total_frames=%d",
        len(_scene_state.all_tracked), _scene_state.total_frames,
    )
    return {
        **_trace("update_memory"),
        "scene_description": _scene_state.get_description(),
        "scene_summary": _scene_state.get_summary(),
    }


def node_detect_change(state: dict[str, Any]) -> dict[str, Any]:
    """Assess whether this frame produced notable events.

    Sets ``has_notable_change`` and ``change_severity`` which downstream
    routing uses to decide alert creation and reasoning triggers.
    """
    new_events = state.get("new_events", [])
    if not new_events:
        logger.debug("[detect_change] no events → severity=none")
        return {
            **_trace("detect_change"),
            "has_notable_change": False,
            "change_severity": "none",
        }

    severities = {e.get("severity", "info") for e in new_events}
    if "alert" in severities:
        sev = "alert"
    elif "warning" in severities:
        sev = "warning"
    else:
        sev = "info"

    logger.debug(
        "[detect_change] %d events, max_severity=%s", len(new_events), sev,
    )
    return {
        **_trace("detect_change"),
        "has_notable_change": True,
        "change_severity": sev,
    }


def node_create_alert(state: dict[str, Any]) -> dict[str, Any]:
    """Promote warning/alert-severity events into the accumulated alerts list."""
    existing = list(state.get("alerts", []))
    new_alerts = [
        e for e in state.get("new_events", [])
        if e.get("severity") in ("warning", "alert")
    ]
    existing.extend(new_alerts)
    logger.debug(
        "[create_alert] +%d alerts, total=%d", len(new_alerts), len(existing),
    )
    return {**_trace("create_alert"), "alerts": existing}


def node_decide_reasoning(state: dict[str, Any]) -> dict[str, Any]:
    """Decide whether LLM reasoning is needed and which task to run.

    Triggers on:
      - explicit ``reasoning_task`` or ``user_question``
      - warning/alert-severity change (→ anomaly reasoning)
      - periodic event-count threshold crossing (image/video only)
    Suppresses periodic summaries in **live** mode for latency.
    """
    global _last_llm_summary_bucket
    mode = state.get("mode", "image")

    # Explicit task or question → always call LLM
    if state.get("reasoning_task") or state.get("user_question"):
        logger.debug("[decide_reasoning] explicit task/question → llm_needed")
        return {**_trace("decide_reasoning"), "llm_needed": True}

    # Warning/alert events → anomaly reasoning
    severity = state.get("change_severity", "none")
    if severity in ("warning", "alert"):
        logger.debug("[decide_reasoning] severity=%s → anomaly", severity)
        return {
            **_trace("decide_reasoning"),
            "llm_needed": True,
            "reasoning_task": "anomaly",
        }

    # Live mode: suppress periodic summaries to keep latency low
    if mode == "live":
        logger.debug("[decide_reasoning] live mode → suppressed")
        return {**_trace("decide_reasoning"), "llm_needed": False}

    # Periodic summary: trigger when crossing a new threshold bucket
    total = state.get("total_event_count", 0)
    threshold = _cfg.llm_trigger_threshold
    summary_bucket = total // threshold if threshold > 0 else 0
    if total > 0 and summary_bucket > _last_llm_summary_bucket:
        _last_llm_summary_bucket = summary_bucket
        logger.debug(
            "[decide_reasoning] bucket %d → summarize", summary_bucket,
        )
        return {
            **_trace("decide_reasoning"),
            "llm_needed": True,
            "reasoning_task": "summarize",
        }

    logger.debug("[decide_reasoning] no trigger → skip")
    return {**_trace("decide_reasoning"), "llm_needed": False}


def node_call_llm(state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the LLM/VLM reasoner for the active task."""
    if _reasoner is None:
        logger.debug("[call_llm] no reasoner configured")
        return {
            **_trace("call_llm"),
            "llm_response": "(No LLM configured)",
            "error": "Select a provider and model first",
        }

    task = state.get("reasoning_task", "describe")
    logger.debug("[call_llm] task=%s", task)

    frame = state.get("current_frame")
    scene = state.get("scene_description", "")
    events_txt = state.get("event_history_text", "")
    det_sum = state.get("detection_summary", "")
    q = state.get("user_question", "")
    oc = json.dumps(state.get("object_counts", {}))

    try:
        if task == "qa" and q:
            resp = _reasoner.answer_question(
                q, image=frame, scene_description=scene,
                recent_events=events_txt, detection_data=det_sum,
            )
        elif task == "summarize":
            resp = _reasoner.summarize_events(
                events_text=events_txt, scene_state=scene,
            )
        elif task == "anomaly":
            resp = _reasoner.reason_anomalies(
                detection_data=det_sum, events_text=events_txt,
                scene_description=scene,
            )
        elif task == "report":
            resp = _reasoner.generate_report(
                session_id=state.get("source_id", "session"),
                duration=state.get("duration", "active"),
                total_frames=state.get("scene_summary", {}).get("total_frames", 0),
                scene_summary=scene, events_text=events_txt, object_stats=oc,
            )
        elif task == "alert":
            alerts = state.get("alerts", [])
            if alerts:
                a = alerts[-1]
                resp = _reasoner.explain_alert(
                    alert_type=a.get("event_type", "?"),
                    severity=a.get("severity", "info"),
                    description=a.get("description", ""),
                    scene_context=scene, recent_events=events_txt,
                )
            else:
                resp = "No alerts to explain."
        else:
            resp = _reasoner.describe_scene(
                image=frame, detection_summary=det_sum,
                object_counts=oc, tracked_objects=scene,
            )
    except Exception as exc:
        logger.error("[call_llm] LLM error: %s", exc)
        return {**_trace("call_llm"), "llm_response": "", "error": f"LLM error: {exc}"}

    result: dict[str, Any] = {**_trace("call_llm"), "llm_response": resp}
    if task == "qa":
        result["answer"] = resp
    if task == "report":
        result["report"] = resp
    logger.debug("[call_llm] response_len=%d", len(resp))
    return result


def node_finalize(state: dict[str, Any]) -> dict[str, Any]:
    """Terminal node — log the completed graph execution path."""
    trace = state.get("_node_trace", [])
    mode = state.get("mode", "?")
    fi = state.get("frame_index", 0)
    n_det = len(state.get("detections", []))
    n_evt = len(state.get("new_events", []))
    llm = state.get("llm_needed", False)
    err = state.get("error", "")
    path = " → ".join(trace + ["finalize"])

    logger.debug(
        "[finalize] mode=%s frame=%d dets=%d events=%d llm=%s err=%r path=[%s]",
        mode, fi, n_det, n_evt, llm, err, path,
    )
    return _trace("finalize")


# ======================================================================
# Q&A-specific nodes
# ======================================================================

def node_qa_cv(state: dict[str, Any]) -> dict[str, Any]:
    """Run CV only if a frame and detector are available; otherwise skip."""
    if state.get("current_frame") is not None and _detector is not None:
        cv_out = node_run_cv(state)
        cv_out["cv_ran"] = True
        # Replace the trace entry — node_run_cv adds its own, override with ours
        cv_out["_node_trace"] = ["qa_cv"]
        logger.debug("[qa_cv] ran CV for Q&A context")
        return cv_out

    logger.debug("[qa_cv] no frame or detector — skipping CV")
    return {**_trace("qa_cv"), "cv_ran": False}


# ======================================================================
# Report-specific nodes
# ======================================================================

def node_gather_report_context(state: dict[str, Any]) -> dict[str, Any]:
    """Assemble current session data for report generation."""
    logger.debug(
        "[gather_report_context] timeline=%d events, tracked=%d",
        _event_timeline.count, len(_scene_state.all_tracked),
    )
    return {
        **_trace("gather_report_context"),
        "object_counts": _scene_state.current_counts,
        "event_history_text": _event_timeline.to_text(),
        "total_event_count": _event_timeline.count,
        "scene_description": _scene_state.get_description(),
        "scene_summary": _scene_state.get_summary(),
    }
