"""LangGraph orchestration — public API for the Agentic Vision Intelligence Platform.

Provides three compiled workflows:

  * **Perception** — full frame-processing pipeline (ingest → CV → events →
    memory → change detection → optional alert → optional LLM → finalize)
  * **Q&A** — user question answering with optional CV enrichment
  * **Report** — session report generation with context gathering

All workflows use a typed :class:`PipelineState` schema and emit debug
traces to the ``src.orchestration`` logger namespace.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    # State
    "PipelineState", "empty_state",
    # Singleton management
    "set_detector", "set_reasoner", "clear_reasoner",
    "get_scene_state", "get_event_timeline", "get_reasoner",
    "reset_session",
    # High-level invocation
    "process_frame", "ask_question", "generate_report",
    # Node functions
    "node_ingest", "node_run_cv", "node_extract_events", "node_update_memory",
    "node_detect_change", "node_create_alert", "node_decide_reasoning",
    "node_call_llm", "node_finalize", "node_qa_cv", "node_gather_report_context",
    # Graph builders & routing
    "build_perception_graph", "build_qa_graph", "build_report_graph",
    "get_perception_wf", "get_qa_wf", "get_report_wf",
    "route_after_ingest", "route_after_cv", "route_after_change",
    "route_after_reasoning", "route_qa_after_cv",
]

# ---- State -----------------------------------------------------------
# ---- Graph builders & routing ----------------------------------------
from src.orchestration.graph import (  # noqa: F401
    build_perception_graph,
    build_qa_graph,
    build_report_graph,
    get_perception_wf,
    get_qa_wf,
    get_report_wf,
    route_after_change,
    route_after_cv,
    route_after_ingest,
    route_after_reasoning,
    route_qa_after_cv,
)

# ---- Singleton management (set from UI) ------------------------------
# ---- Node functions (re-exported for direct testing) -----------------
from src.orchestration.nodes import (  # noqa: F401  # noqa: F401
    # helpers (used by tests / advanced callers)
    _dets_from_state,
    _frame_result_from_state,
    clear_reasoner,
    get_event_timeline,
    get_reasoner,
    get_scene_state,
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
    set_detector,
)
from src.orchestration.nodes import (
    reset_all as _reset_all_singletons,
)
from src.orchestration.nodes import (
    set_reasoner_obj as set_reasoner,
)
from src.orchestration.state import PipelineState, empty_state  # noqa: F401

# ======================================================================
# CROSS-FRAME STATE
# ======================================================================
#
# ``last_llm_summary_bucket`` lives on PipelineState so that every node is a
# pure function of its input.  The "once per bucket" gate, however, needs to
# survive across successive frames — each frame is a fresh graph invocation
# with a fresh state.  A tiny module-scoped cache bridges frames without
# adding any cross-session coupling.  It is reset by ``reset_session()``.

_summary_bucket_cache: list[int] = [0]


def reset_session() -> None:
    """Reset every session-scoped singleton and cross-frame cache.

    Wraps :func:`_reset_all_singletons` so both the detector/reasoner/memory
    singletons *and* the cross-frame summary-bucket counter clear together.
    """
    _reset_all_singletons()
    _summary_bucket_cache[0] = 0


# ======================================================================
# HIGH-LEVEL INVOCATION
# ======================================================================

def process_frame(
    frame,
    *,
    mode: str = "image",
    source_id: str = "",
    frame_index: int = 0,
    draw: bool = True,
) -> dict[str, Any]:
    """Run the full perception pipeline on a single frame.

    ``draw=False`` suppresses annotation rendering in ``node_run_cv`` when
    the caller will not display the annotated frame (e.g. non-display ticks
    in video mode); detections and counts are still produced.
    """
    s = {
        **empty_state(),
        "mode": mode,
        "current_frame": frame,
        "source_id": source_id,
        "frame_index": frame_index,
        "draw": draw,
        # Carry forward the last-fired summary bucket across frames.
        "last_llm_summary_bucket": _summary_bucket_cache[0],
    }
    out = get_perception_wf().invoke(s)
    _summary_bucket_cache[0] = out.get("last_llm_summary_bucket", _summary_bucket_cache[0])
    return out


def ask_question(
    question: str, *, frame=None, mode: str = "image"
) -> dict[str, Any]:
    """Answer a user question, optionally enriched with CV context.

    Q&A is read-only — it never mutates scene memory or the event timeline.
    All context is pre-populated here from the current session state.
    """
    from src.orchestration.nodes import get_event_timeline, get_scene_state

    tl = get_event_timeline()
    ss = get_scene_state()
    s = {
        **empty_state(),
        "mode": mode,
        "current_frame": frame,
        "llm_needed": True,
        "reasoning_task": "qa",
        "user_question": question,
        "event_history_text": tl.to_text(),
        "total_event_count": tl.count,
        "scene_description": ss.get_description(),
        "scene_summary": ss.get_summary(),
    }
    return get_qa_wf().invoke(s)


def generate_report(
    *, session_id: str = "session", duration: str = "active"
) -> dict[str, Any]:
    """Generate a session report — context is gathered by the graph itself."""
    s = {
        **empty_state(),
        "llm_needed": True,
        "reasoning_task": "report",
        "source_id": session_id,
        "duration": duration,
    }
    return get_report_wf().invoke(s)

