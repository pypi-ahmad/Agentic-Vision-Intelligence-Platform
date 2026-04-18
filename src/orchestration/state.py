"""Pipeline state schema — TypedDict used by LangGraph StateGraph.

Every workflow (perception, Q&A, report) shares this single state schema.
Fields are typed so LangGraph can enforce channel semantics.  ``_node_trace``
uses an append-reducer so each node *adds* to the list rather than replacing
it — giving free debug visibility of the path taken through the graph.
"""

import operator
from typing import Annotated, Any, TypedDict


class PipelineState(TypedDict, total=False):
    """Typed graph state for all orchestration workflows.

    ``total=False`` — nodes return only the fields they modify;
    LangGraph merges each update into the running state via
    *last-writer-wins* (default) or the annotated reducer.
    """

    # ---- Input context -----------------------------------------------
    mode: str                               # "image" | "video" | "live"
    current_frame: Any                      # np.ndarray (BGR) or None
    source_id: str                          # media source identifier
    frame_index: int                        # sequential frame number

    # ---- CV detection results ----------------------------------------
    detections: list[dict[str, Any]]        # serialised Detection dicts
    object_counts: dict[str, int]           # {class_name: count}
    detection_summary: str                  # human-readable one-liner
    annotated_frame: Any                    # np.ndarray with overlays or None
    draw: bool                              # whether node_run_cv should annotate

    # ---- Scene memory ------------------------------------------------
    scene_description: str                  # rolling description from SceneState
    scene_summary: dict[str, Any]           # structured summary dict

    # ---- Event timeline ----------------------------------------------
    new_events: list[dict[str, Any]]        # events extracted this frame
    event_history_text: str                 # full timeline as text
    total_event_count: int                  # cumulative event count

    # ---- Change detection --------------------------------------------
    has_notable_change: bool                # True if this frame produced events
    change_severity: str                    # "none" | "info" | "warning" | "alert"

    # ---- Reasoning / LLM --------------------------------------------
    llm_needed: bool                        # whether LLM should be called
    reasoning_task: str                     # "" | "describe" | "summarize" | "anomaly" | "qa" | "report" | "alert"
    llm_response: str                       # raw LLM output
    last_llm_summary_bucket: int            # last event-count bucket that triggered a summary

    # ---- Q&A ---------------------------------------------------------
    user_question: str                      # user's question text
    answer: str                             # LLM answer for Q&A

    # ---- Alerts ------------------------------------------------------
    alerts: list[dict[str, Any]]            # accumulated alert events

    # ---- Reports -----------------------------------------------------
    report: str                             # generated report text
    duration: str                           # session duration string

    # ---- Internal flow control ---------------------------------------
    cv_ran: bool                            # whether CV was executed this pass
    error: str                              # error message, "" if none

    # ---- Debug tracing (append-only) ---------------------------------
    _node_trace: Annotated[list[str], operator.add]


def empty_state() -> dict[str, Any]:
    """Return a clean initial state dict with all fields at default values."""
    return dict(
        mode="image",
        current_frame=None,
        source_id="",
        frame_index=0,
        detections=[],
        object_counts={},
        detection_summary="",
        annotated_frame=None,
        draw=True,
        scene_description="",
        scene_summary={},
        new_events=[],
        event_history_text="",
        total_event_count=0,
        has_notable_change=False,
        change_severity="none",
        llm_needed=False,
        reasoning_task="",
        llm_response="",
        last_llm_summary_bucket=0,
        user_question="",
        answer="",
        alerts=[],
        report="",
        duration="",
        cv_ran=False,
        error="",
        _node_trace=[],
    )
