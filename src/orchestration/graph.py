"""LangGraph workflow definitions — graph builders, routing, and compiled cache.

Three workflows are defined:

* **Perception** — full frame-processing pipeline (9 nodes, 3 conditional edges)
* **Q&A** — user question answering with optional CV enrichment
* **Report** — session report generation with context gathering
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.orchestration.state import PipelineState
from src.orchestration.nodes import (
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

logger = logging.getLogger(__name__)


# ======================================================================
# Routing functions
# ======================================================================

def route_after_ingest(state: dict[str, Any]) -> str:
    """Short-circuit to finalize if ingest detected an error (e.g. missing frame)."""
    if state.get("error"):
        return "finalize"
    return "run_cv"


def route_after_change(state: dict[str, Any]) -> str:
    """Route based on change severity: warnings/alerts → create_alert first."""
    sev = state.get("change_severity", "none")
    if sev in ("warning", "alert"):
        return "create_alert"
    return "decide_reasoning"


def route_after_reasoning(state: dict[str, Any]) -> str:
    """Call the LLM only when reasoning was deemed necessary."""
    return "call_llm" if state.get("llm_needed") else "finalize"


def route_qa_after_cv(state: dict[str, Any]) -> str:
    """In Q&A workflow: update memory if CV ran, otherwise skip to LLM."""
    return "update_memory" if state.get("cv_ran") else "call_llm"


# ======================================================================
# Graph Builders
# ======================================================================

def build_perception_graph() -> StateGraph:
    """Main perception pipeline::

        ingest ─┬─► run_cv → extract_events → update_memory → detect_change
                │                                                  │
                │                                    ┌─────────────┼────────────┐
                │                                    ▼ (warn/alert)             ▼ (info/none)
                │                               create_alert ──► decide_reasoning
                │                                                      │
                │                                       ┌──────────────┼───────────┐
                │                                       ▼ (llm_needed)             ▼ (skip)
                │                                   call_llm ──► finalize ◄────────┘
                │
                └─► finalize  (ingest error short-circuit)
    """
    g = StateGraph(PipelineState)

    g.add_node("ingest", node_ingest)
    g.add_node("run_cv", node_run_cv)
    g.add_node("extract_events", node_extract_events)
    g.add_node("update_memory", node_update_memory)
    g.add_node("detect_change", node_detect_change)
    g.add_node("create_alert", node_create_alert)
    g.add_node("decide_reasoning", node_decide_reasoning)
    g.add_node("call_llm", node_call_llm)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("ingest")

    g.add_conditional_edges("ingest", route_after_ingest, {
        "run_cv": "run_cv",
        "finalize": "finalize",
    })
    g.add_edge("run_cv", "extract_events")
    g.add_edge("extract_events", "update_memory")
    g.add_edge("update_memory", "detect_change")
    g.add_conditional_edges("detect_change", route_after_change, {
        "create_alert": "create_alert",
        "decide_reasoning": "decide_reasoning",
    })
    g.add_edge("create_alert", "decide_reasoning")
    g.add_conditional_edges("decide_reasoning", route_after_reasoning, {
        "call_llm": "call_llm",
        "finalize": "finalize",
    })
    g.add_edge("call_llm", "finalize")
    g.add_edge("finalize", END)

    return g


def build_qa_graph() -> StateGraph:
    """Q&A workflow::

        qa_cv ─┬─► update_memory → call_llm → finalize
               └─► call_llm → finalize
    """
    g = StateGraph(PipelineState)

    g.add_node("qa_cv", node_qa_cv)
    g.add_node("update_memory", node_update_memory)
    g.add_node("call_llm", node_call_llm)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("qa_cv")
    g.add_conditional_edges("qa_cv", route_qa_after_cv, {
        "update_memory": "update_memory",
        "call_llm": "call_llm",
    })
    g.add_edge("update_memory", "call_llm")
    g.add_edge("call_llm", "finalize")
    g.add_edge("finalize", END)

    return g


def build_report_graph() -> StateGraph:
    """Report generation workflow::

        gather_context → call_llm → finalize
    """
    g = StateGraph(PipelineState)

    g.add_node("gather_context", node_gather_report_context)
    g.add_node("call_llm", node_call_llm)
    g.add_node("finalize", node_finalize)

    g.set_entry_point("gather_context")
    g.add_edge("gather_context", "call_llm")
    g.add_edge("call_llm", "finalize")
    g.add_edge("finalize", END)

    return g


# ======================================================================
# Compiled cache
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
