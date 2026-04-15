"""Shared test fixtures."""

import numpy as np
import pytest

from src.vision.detector import Detection, FrameResult
from src.vision.events import SceneEvent


@pytest.fixture
def sample_frame():
    """A 100x100 black RGB frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    return Detection(class_id=0, class_name="person", confidence=0.9, bbox=(10, 20, 50, 60))


@pytest.fixture
def sample_detection_tracked():
    return Detection(class_id=0, class_name="person", confidence=0.9, bbox=(10, 20, 50, 60), track_id=1)


@pytest.fixture
def sample_frame_result():
    dets = [
        Detection(0, "person", 0.9, (10, 20, 50, 60), track_id=1),
        Detection(1, "car", 0.8, (100, 100, 200, 200), track_id=2),
    ]
    return FrameResult(frame_index=0, detections=dets, object_counts={"person": 1, "car": 1})


@pytest.fixture
def sample_event():
    return SceneEvent(event_type="object_appeared", description="A person appeared", severity="info")


@pytest.fixture
def sample_warning_event():
    return SceneEvent(event_type="crowding", description="High density", severity="warning")
