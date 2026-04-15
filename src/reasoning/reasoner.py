"""High-level reasoning interface — wraps any LLMProvider."""

from __future__ import annotations

import logging

import numpy as np

from src.providers.base import LLMProvider
from src.reasoning import prompts

logger = logging.getLogger(__name__)


class Reasoner:
    """Task-oriented reasoning that delegates to the active LLMProvider."""

    def __init__(self, provider: LLMProvider, model: str):
        self._prov = provider
        self._model = model

    def _call(self, prompt: str, *, system: str | None = None,
              images: list[np.ndarray] | None = None) -> str:
        return self._prov.generate(prompt, model=self._model, images=images, system=system)

    # ---- tasks -------------------------------------------------------

    def describe_scene(self, *, image: np.ndarray | None = None,
                       detection_summary: str = "", object_counts: str = "",
                       tracked_objects: str = "") -> str:
        p = prompts.SCENE_DESCRIPTION.format(
            detection_summary=detection_summary or "N/A",
            object_counts=object_counts or "N/A",
            tracked_objects=tracked_objects or "N/A",
        )
        return self._call(p, system=prompts.SYSTEM_ANALYST, images=[image] if image is not None else None)

    def summarize_events(self, *, events_text: str, scene_state: str) -> str:
        p = prompts.EVENT_SUMMARY.format(events_text=events_text or "None.", scene_state=scene_state or "N/A")
        return self._call(p, system=prompts.SYSTEM_ANALYST)

    def answer_question(self, question: str, *, image: np.ndarray | None = None,
                        scene_description: str = "", recent_events: str = "",
                        detection_data: str = "") -> str:
        p = prompts.QUESTION_ANSWER.format(
            scene_description=scene_description or "N/A",
            recent_events=recent_events or "None.",
            detection_data=detection_data or "N/A",
            question=question,
        )
        return self._call(p, system=prompts.SYSTEM_QA, images=[image] if image is not None else None)

    def generate_report(self, *, session_id: str, duration: str,
                        total_frames: int, scene_summary: str,
                        events_text: str, object_stats: str) -> str:
        p = prompts.INCIDENT_REPORT.format(
            session_id=session_id, duration=duration, total_frames=total_frames,
            scene_summary=scene_summary or "N/A",
            events_text=events_text or "None.",
            object_stats=object_stats or "N/A",
        )
        return self._call(p, system=prompts.SYSTEM_REPORT)

    def reason_anomalies(self, *, detection_data: str, events_text: str,
                         scene_description: str) -> str:
        p = prompts.ANOMALY_REASONING.format(
            detection_data=detection_data or "N/A",
            events_text=events_text or "None.",
            scene_description=scene_description or "N/A",
        )
        return self._call(p, system=prompts.SYSTEM_ANALYST)

    def explain_alert(self, *, alert_type: str, severity: str, description: str,
                      scene_context: str, recent_events: str) -> str:
        p = prompts.ALERT_EXPLANATION.format(
            alert_type=alert_type, severity=severity, description=description,
            scene_context=scene_context or "N/A",
            recent_events=recent_events or "None.",
        )
        return self._call(p, system=prompts.SYSTEM_ANALYST)
