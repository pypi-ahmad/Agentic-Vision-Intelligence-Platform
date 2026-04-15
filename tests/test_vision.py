"""Tests for the vision layer — Detection, FrameResult, VisionDetector."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.vision.detector import Detection, FrameResult


class TestDetection:
    def test_center(self, sample_detection):
        assert sample_detection.center == (30, 40)

    def test_area(self):
        d = Detection(class_id=0, class_name="car", confidence=0.8, bbox=(0, 0, 100, 50))
        assert d.area == 5000

    def test_to_dict_without_track(self):
        d = Detection(class_id=1, class_name="dog", confidence=0.75, bbox=(1, 2, 3, 4))
        out = d.to_dict()
        assert out["class_name"] == "dog"
        assert "track_id" not in out

    def test_to_dict_with_track(self, sample_detection_tracked):
        assert sample_detection_tracked.to_dict()["track_id"] == 1

    def test_zero_area(self):
        d = Detection(class_id=0, class_name="x", confidence=0.5, bbox=(5, 5, 5, 5))
        assert d.area == 0


class TestFrameResult:
    def test_summary_line_empty(self):
        fr = FrameResult(frame_index=0)
        assert fr.summary_line == "no objects detected"

    def test_summary_line_counts(self):
        fr = FrameResult(frame_index=1, object_counts={"person": 3, "car": 1})
        assert "person" in fr.summary_line
        assert "car" in fr.summary_line

    def test_to_dict(self):
        d = Detection(0, "a", 0.9, (0, 0, 1, 1))
        fr = FrameResult(frame_index=5, detections=[d], object_counts={"a": 1})
        out = fr.to_dict()
        assert out["frame_index"] == 5
        assert out["num_detections"] == 1


class TestVisionDetector:
    @patch("src.vision.detector.load_yolo")
    @patch("src.vision.detector.get_settings")
    def test_detect_calls_model(self, mock_settings, mock_load):
        mock_cfg = MagicMock()
        mock_cfg.yolo_confidence = 0.4
        mock_cfg.resolve_device.return_value = "cpu"
        mock_settings.return_value = mock_cfg

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.names = {}
        mock_result.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_model.predict.return_value = [mock_result]
        mock_load.return_value = mock_model

        from src.vision.detector import VisionDetector
        det = VisionDetector(variant="YOLO26n")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert isinstance(result, FrameResult)
        mock_model.predict.assert_called_once()

    @patch("src.vision.detector.load_yolo")
    @patch("src.vision.detector.get_settings")
    def test_confidence_zero_is_respected(self, mock_settings, mock_load):
        """Explicit confidence=0.0 should NOT fall back to config default."""
        mock_cfg = MagicMock()
        mock_cfg.yolo_confidence = 0.5
        mock_cfg.resolve_device.return_value = "cpu"
        mock_settings.return_value = mock_cfg
        mock_load.return_value = MagicMock()

        from src.vision.detector import VisionDetector
        det = VisionDetector(variant="YOLO26n", confidence=0.0)
        assert det._conf == 0.0  # not 0.5

    @patch("src.vision.detector.load_yolo")
    @patch("src.vision.detector.get_settings")
    def test_detect_with_zero_confidence_passes_zero(self, mock_settings, mock_load):
        """detect(confidence=0.0) must forward 0.0 to the model, not the default."""
        mock_cfg = MagicMock()
        mock_cfg.yolo_confidence = 0.5
        mock_cfg.resolve_device.return_value = "cpu"
        mock_settings.return_value = mock_cfg

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.names = {}
        mock_result.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_model.predict.return_value = [mock_result]
        mock_load.return_value = mock_model

        from src.vision.detector import VisionDetector
        det = VisionDetector(variant="YOLO26n", confidence=0.4)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det.detect(frame, confidence=0.0)
        call_kwargs = mock_model.predict.call_args
        assert call_kwargs[1]["conf"] == 0.0  # not 0.4

    @patch("src.vision.detector.load_yolo")
    @patch("src.vision.detector.get_settings")
    def test_track_with_zero_confidence_passes_zero(self, mock_settings, mock_load):
        """track(confidence=0.0) must forward 0.0 to the model, not the default."""
        mock_cfg = MagicMock()
        mock_cfg.yolo_confidence = 0.5
        mock_cfg.resolve_device.return_value = "cpu"
        mock_settings.return_value = mock_cfg

        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_result.names = {}
        mock_result.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_model.track.return_value = [mock_result]
        mock_load.return_value = mock_model

        from src.vision.detector import VisionDetector
        det = VisionDetector(variant="YOLO26n", confidence=0.4)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det.track(frame, confidence=0.0)
        call_kwargs = mock_model.track.call_args
        assert call_kwargs[1]["conf"] == 0.0

    @patch("src.vision.detector.load_yolo")
    @patch("src.vision.detector.get_settings")
    def test_variant_property(self, mock_settings, mock_load):
        mock_cfg = MagicMock()
        mock_cfg.yolo_confidence = 0.4
        mock_cfg.resolve_device.return_value = "cpu"
        mock_settings.return_value = mock_cfg
        mock_load.return_value = MagicMock()

        from src.vision.detector import VisionDetector
        det = VisionDetector(variant="YOLO26l")
        assert det.variant == "YOLO26l"
