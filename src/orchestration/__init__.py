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
    "route_after_ingest", "route_after_change",
    "route_after_reasoning", "route_qa_after_cv",
]

# ---- State -----------------------------------------------------------
from src.orchestration.state import PipelineState, empty_state  # noqa: F401

# ---- Singleton management (set from UI) ------------------------------
from src.orchestration.nodes import (  # noqa: F401
    set_detector,
    set_reasoner_obj as set_reasoner,
    clear_reasoner,
    get_scene_state,
    get_event_timeline,
    get_reasoner,
    reset_all as reset_session,
    # helpers (used by tests / advanced callers)
    _dets_from_state,
    _frame_result_from_state,
    # expose singletons for test assertions
    _scene_state,
)

# ---- Node functions (re-exported for direct testing) -----------------
from src.orchestration.nodes import (  # noqa: F401
    node_ingest,
    node_run_cv,
    node_extract_events,
    node_update_memory,
    node_detect_change,
    node_create_alert,
    node_decide_reasoning,
    node_call_llm,
    node_finalize,
    node_qa_cv,
    node_gather_report_context,
)

# ---- Graph builders & routing ----------------------------------------
from src.orchestration.graph import (  # noqa: F401
    build_perception_graph,
    build_qa_graph,
    build_report_graph,
    get_perception_wf,
    get_qa_wf,
    get_report_wf,
    route_after_ingest,
    route_after_change,
    route_after_reasoning,
    route_qa_after_cv,
)

# ---- Backward-compatibility aliases ---------------------------------
node_decide_llm = node_decide_reasoning  # old name used in existing tests
_node_qa_cv = node_qa_cv                 # old private name


def _empty_state() -> dict:
    """Backward-compat alias — returns a fresh copy every time."""
    return empty_state()


# Kept as a property-like callable for old code that does {**_EMPTY_STATE, ...}
_EMPTY_STATE = empty_state()


# ======================================================================
# HIGH-LEVEL INVOCATION
# ======================================================================

def process_frame(
    frame,
    *,
    mode: str = "image",
    source_id: str = "",
    frame_index: int = 0,
) -> dict[str, Any]:
    """Run the full perception pipeline on a single frame."""
    s = {
        **empty_state(),
        "mode": mode,
        "current_frame": frame,
        "source_id": source_id,
        "frame_index": frame_index,
    }
    return get_perception_wf().invoke(s)


def ask_question(
    question: str, *, frame=None, mode: str = "image"
) -> dict[str, Any]:
    """Answer a user question, optionally enriched with CV context."""
    from src.orchestration.nodes import _event_timeline, _scene_state as _ss

    s = {
        **empty_state(),
        "mode": mode,
        "current_frame": frame,
        "llm_needed": True,
        "reasoning_task": "qa",
        "user_question": question,
        "event_history_text": _event_timeline.to_text(),
        "total_event_count": _event_timeline.count,
        "scene_description": _ss.get_description(),
        "scene_summary": _ss.get_summary(),
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

