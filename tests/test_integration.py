"""Integration tests for orchestration workflows and Reasoner dispatch.

These tests cover hot paths that were previously exercised only by unit tests:

* end-to-end perception graph invocation with a stub provider
* Q&A graph round-trip — must *not* mutate scene memory (C2 regression)
* perception short-circuits to ``finalize`` when CV fails (C1 regression)
* Reasoner task methods reach the provider with the expected prompt + images
"""

from __future__ import annotations

import numpy as np
import pytest

from src.orchestration import (
    ask_question,
    empty_state,
    get_perception_wf,
    get_qa_wf,
    process_frame,
    reset_session,
    set_detector,
    set_reasoner,
)
from src.orchestration import nodes as _nodes
from src.providers.base import LLMProvider
from src.reasoning.reasoner import Reasoner

# ======================================================================
# Stub provider / detector used across this module
# ======================================================================


class StubProvider(LLMProvider):
    name = "Stub"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def list_models(self) -> list[str]:
        return ["stub-model"]

    def generate(self, prompt, *, model, images=None, system=None):
        self.calls.append(
            {"prompt": prompt, "model": model, "images": images or [], "system": system}
        )
        return f"stub-response:{model}"


class StubDetector:
    variant = "stub"

    def detect(self, frame, *, confidence=None, draw=True):
        from src.vision.detector import Detection, FrameResult

        dets = [Detection(0, "person", 0.9, (10, 10, 50, 50))]
        ann = np.zeros_like(frame) if draw else None
        return FrameResult(
            frame_index=0,
            detections=dets,
            object_counts={"person": 1},
            annotated_frame=ann,
        )

    def track(self, frame, *, confidence=None, persist=True, draw=True):
        return self.detect(frame, draw=draw)


@pytest.fixture
def stub_detector(monkeypatch):
    reset_session()
    det = StubDetector()
    monkeypatch.setattr(_nodes, "_detector", det)
    yield det
    reset_session()


@pytest.fixture
def stub_reasoner_with_provider():
    reset_session()
    prov = StubProvider()
    set_reasoner(prov, "stub-model")
    yield prov
    reset_session()


# ======================================================================
# End-to-end perception graph
# ======================================================================


class TestPerceptionGraph:
    def test_happy_path_runs_all_stages(self, stub_detector):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = process_frame(frame, mode="image", source_id="t", frame_index=0)

        assert out["error"] == ""
        assert out["object_counts"] == {"person": 1}
        assert out["annotated_frame"] is not None
        # Full path must include finalize after real CV.
        trace = out["_node_trace"]
        assert "ingest" in trace and "run_cv" in trace and "finalize" in trace

    def test_cv_error_short_circuits_to_finalize(self):
        """C1 regression: no detector => error, downstream nodes skipped."""
        reset_session()  # ensure no detector
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = process_frame(frame, mode="image")

        assert out["error"]
        # downstream memory/event nodes must NOT run
        trace = out["_node_trace"]
        assert "extract_events" not in trace
        assert "update_memory" not in trace
        assert "detect_change" not in trace

        # scene state must not have been polluted
        scene = _nodes.get_scene_state()
        assert scene.total_frames == 0

    def test_draw_false_skips_annotation(self, stub_detector):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = process_frame(frame, mode="video", draw=False)
        assert out["annotated_frame"] is None
        # detections are still produced
        assert out["object_counts"] == {"person": 1}


# ======================================================================
# Q&A graph — must be read-only
# ======================================================================


class TestQAGraph:
    def test_qa_does_not_mutate_scene(self, stub_detector, stub_reasoner_with_provider):
        """C2 regression: Q&A must never call SceneState.update()."""
        # Seed scene state via a real perception run.
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        process_frame(frame, mode="image")
        scene = _nodes.get_scene_state()
        frames_before = scene.total_frames
        counts_before = dict(scene.current_counts)

        res = ask_question("How many people?", frame=frame, mode="image")
        assert res.get("answer", "").startswith("stub-response:")

        assert scene.total_frames == frames_before
        assert scene.current_counts == counts_before

    def test_qa_without_detector_still_calls_llm(self, stub_reasoner_with_provider):
        """Q&A must degrade gracefully when CV isn't available."""
        reset_session()
        set_reasoner(stub_reasoner_with_provider, "stub-model")
        res = ask_question("What is going on?")
        assert res.get("answer", "").startswith("stub-response:")


# ======================================================================
# Reasoner dispatch — verify prompts + images reach the provider
# ======================================================================


class TestReasonerDispatch:
    def _fresh(self):
        prov = StubProvider()
        return prov, Reasoner(prov, "stub-model")

    def test_describe_scene_passes_image_and_system(self):
        prov, r = self._fresh()
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        r.describe_scene(image=img, detection_summary="1 person")

        assert len(prov.calls) == 1
        call = prov.calls[0]
        assert "1 person" in call["prompt"]
        assert call["system"] is not None
        assert len(call["images"]) == 1

    def test_answer_question_includes_question_text(self):
        prov, r = self._fresh()
        r.answer_question("Where is the dog?", scene_description="room")
        assert "Where is the dog?" in prov.calls[0]["prompt"]

    def test_generate_report_uses_report_system(self):
        prov, r = self._fresh()
        r.generate_report(
            session_id="s1", duration="10m", total_frames=100,
            scene_summary="stable", events_text="none", object_stats="{}",
        )
        assert "s1" in prov.calls[0]["prompt"]
        # Report system prompt mentions "report writer"
        assert "report" in (prov.calls[0]["system"] or "").lower()

    def test_reason_anomalies_no_image(self):
        prov, r = self._fresh()
        r.reason_anomalies(
            detection_data="1 person",
            events_text="crowding",
            scene_description="room",
        )
        assert prov.calls[0]["images"] == []

    def test_explain_alert_formats_all_fields(self):
        prov, r = self._fresh()
        r.explain_alert(
            alert_type="crowding",
            severity="warning",
            description="density high",
            scene_context="lobby",
            recent_events="...",
        )
        p = prov.calls[0]["prompt"]
        assert "crowding" in p
        assert "warning" in p
        assert "density high" in p
