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

from config import MODE_USES_TRACKING, get_settings
from src.memory.event_timeline import EventTimeline
from src.memory.scene_state import SceneState
from src.providers.base import LLMProvider
from src.reasoning.reasoner import Reasoner
from src.vision.detector import Detection, FrameResult, VisionDetector
from src.vision.events import EventExtractor

logger = logging.getLogger(__name__)


# ======================================================================
# Session-scoped singletons (reset via reset_all)
#
# Singletons are lazily initialised on first use so that ``.env`` changes
# take effect without a process restart. ``_last_llm_summary_bucket`` was
# moved onto PipelineState (see state.py) — nodes now read/write it via the
# graph's typed state instead of mutating a module global.
# ======================================================================

_event_extractor: EventExtractor | None = None
_scene_state: SceneState | None = None
_event_timeline: EventTimeline | None = None

_detector: VisionDetector | None = None
_reasoner: Reasoner | None = None


def _ensure_session() -> None:
    """Lazily construct session-scoped singletons from current settings."""
    global _event_extractor, _scene_state, _event_timeline
    if _event_extractor is None or _scene_state is None or _event_timeline is None:
        cfg = get_settings()
        if _event_extractor is None:
            _event_extractor = EventExtractor(cooldown_seconds=cfg.event_cooldown_seconds)
        if _scene_state is None:
            _scene_state = SceneState(window_seconds=cfg.memory_window_seconds)
        if _event_timeline is None:
            _event_timeline = EventTimeline()


_ensure_session()


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
    _ensure_session()
    return _scene_state  # type: ignore[return-value]


def get_event_timeline() -> EventTimeline:
    _ensure_session()
    return _event_timeline  # type: ignore[return-value]


def get_reasoner() -> Reasoner | None:
    return _reasoner


def reset_all() -> None:
    """Reset session-scoped state — detector, reasoner, memory, timeline, extractor."""
    global _detector, _reasoner
    _ensure_session()
    assert _event_extractor is not None and _scene_state is not None and _event_timeline is not None
    _event_extractor.reset()
    _scene_state.reset()
    _event_timeline.reset()
    _detector = None
    _reasoner = None


# ======================================================================
# Helpers
# ======================================================================

def _trace(name: str) -> dict[str, Any]:
    """Return a partial-state dict that appends *name* to _node_trace."""
    return {"_node_trace": [name]}


def _dets_from_state(state: dict[str, Any]) -> list[Detection]:
    """Reconstruct Detection objects from serialised dicts in graph state."""
    out: list[Detection] = []
    for d in state.get("detections", []):
        bbox_seq = d.get("bbox", [0, 0, 0, 0])
        if len(bbox_seq) != 4:
            raise ValueError(
                f"Detection bbox must have 4 elements, got {len(bbox_seq)}: {bbox_seq!r}"
            )
        bbox: tuple[int, int, int, int] = (
            int(bbox_seq[0]), int(bbox_seq[1]), int(bbox_seq[2]), int(bbox_seq[3]),
        )
        out.append(
            Detection(
                class_id=int(d.get("class_id", 0)),
                class_name=str(d.get("class_name", "?")),
                confidence=float(d.get("confidence", 0.0)),
                bbox=bbox,
                track_id=d.get("track_id"),
            )
        )
    return out


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
    # ``draw`` is sourced from state so callers (process_frame) can suppress
    # the ~5\u201320 ms annotation overhead on frames that will not be rendered.
    draw = bool(state.get("draw", True))
    try:
        if MODE_USES_TRACKING.get(mode, False):
            result = _detector.track(frame, persist=True, draw=draw)
        else:
            result = _detector.detect(frame, draw=draw)
    except Exception as exc:
        # Redact: raw detector/YOLO exceptions may include file paths, weight
        # URLs, or CUDA diagnostics — keep detail in logs, surface a stable
        # category marker to state/UI.
        logger.error("[run_cv] CV inference failed", exc_info=exc)
        return {**_trace("run_cv"), "error": "CV inference failed — see logs for details"}

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
    _ensure_session()
    assert _event_extractor is not None and _event_timeline is not None
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
    _ensure_session()
    assert _scene_state is not None
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

    Suppresses periodic summaries in **live** mode for latency.  The
    "last bucket fired" counter lives on :class:`PipelineState` so the node
    is a pure function of its input — compatible with LangGraph checkpoint
    replay and multi-context execution.
    """
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
    threshold = get_settings().llm_trigger_threshold
    summary_bucket = total // threshold if threshold > 0 else 0
    last_bucket = state.get("last_llm_summary_bucket", 0)
    if total > 0 and summary_bucket > last_bucket:
        logger.debug(
            "[decide_reasoning] bucket %d → summarize", summary_bucket,
        )
        return {
            **_trace("decide_reasoning"),
            "llm_needed": True,
            "reasoning_task": "summarize",
            "last_llm_summary_bucket": summary_bucket,
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
        # S1: log full detail for operators, surface a redacted marker to the
        # UI / downstream consumers so raw SDK messages (URLs, partial bodies,
        # stack fragments) never leak into shared session state.
        logger.error("[call_llm] LLM error", exc_info=exc)
        return {**_trace("call_llm"), "llm_response": "", "error": "LLM error — see logs for details"}

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
    _ensure_session()
    assert _scene_state is not None and _event_timeline is not None
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
