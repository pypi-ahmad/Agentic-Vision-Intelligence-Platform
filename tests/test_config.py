"""Tests for config module."""

import pytest

from config import YOLO_MODELS, MODE_DEFAULT_MODELS, MODE_MAX_DIM, MODE_USES_TRACKING, VIDEO_DISPLAY_INTERVAL, PROVIDERS, Settings, get_settings


class TestSettings:
    def test_default_values(self):
        s = Settings()
        assert s.yolo_confidence == 0.35
        assert s.camera_index == 0
        assert s.frame_sample_rate >= 1
        assert s.memory_window_seconds >= 10
        assert s.event_cooldown_seconds >= 1

    def test_output_path_creates_dir(self, tmp_path):
        s = Settings(report_output_dir=str(tmp_path / "out"))
        p = s.output_path
        assert p.exists()

    def test_resolve_device_cpu_fallback(self):
        s = Settings()
        dev = s.resolve_device()
        assert dev in ("cpu", "0")


class TestConstants:
    def test_yolo_models_keys(self):
        assert set(YOLO_MODELS.keys()) == {"YOLO26n", "YOLO26m", "YOLO26l"}

    def test_yolo_models_values(self):
        for v in YOLO_MODELS.values():
            assert v.endswith(".pt")

    def test_mode_defaults(self):
        assert MODE_DEFAULT_MODELS["live"] == "YOLO26n"
        assert MODE_DEFAULT_MODELS["image"] == "YOLO26l"
        assert MODE_DEFAULT_MODELS["video"] == "YOLO26m"

    def test_mode_defaults_reference_valid_variants(self):
        for mode, variant in MODE_DEFAULT_MODELS.items():
            assert variant in YOLO_MODELS, f"Mode '{mode}' defaults to unknown variant '{variant}'"

    def test_providers_list(self):
        assert PROVIDERS == ["Ollama", "OpenAI", "Gemini", "Anthropic"]

    def test_mode_max_dim_live_is_smallest(self):
        assert MODE_MAX_DIM["live"] <= MODE_MAX_DIM["video"] <= MODE_MAX_DIM["image"]

    def test_mode_max_dim_covers_all_modes(self):
        for mode in MODE_DEFAULT_MODELS:
            assert mode in MODE_MAX_DIM

    def test_mode_uses_tracking(self):
        assert MODE_USES_TRACKING["live"] is True
        assert MODE_USES_TRACKING["video"] is True
        assert MODE_USES_TRACKING["image"] is False

    def test_video_display_interval_positive(self):
        assert VIDEO_DISPLAY_INTERVAL >= 1


class TestGetSettings:
    def test_singleton(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
