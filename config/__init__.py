"""Agentic Vision Intelligence Platform — Central Configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider keys (optional) ---
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama API base URL")
    openai_api_key: str = Field("", description="OpenAI API key")
    gemini_api_key: str = Field("", description="Google Gemini API key")
    anthropic_api_key: str = Field("", description="Anthropic API key")

    # --- YOLO ---
    yolo_confidence: float = Field(0.35, ge=0.0, le=1.0)

    # --- Camera ---
    camera_index: int = Field(0)
    camera_width: int = Field(1280)
    camera_height: int = Field(720)

    # --- Processing ---
    frame_sample_rate: int = Field(5, ge=1)
    memory_window_seconds: int = Field(300, ge=10)
    event_cooldown_seconds: int = Field(10, ge=1)
    llm_trigger_threshold: int = Field(3, ge=1)

    # --- Reporting ---
    report_output_dir: str = Field("output")

    # --- Security ---
    # Allow the Ollama base URL to point at arbitrary remote hosts.  When
    # ``False`` (default), only loopback / private-RFC-1918 targets are
    # accepted \u2014 this prevents SSRF via the Streamlit "Ollama URL" input
    # in hosted deployments.
    allow_remote_ollama: bool = Field(False)

    # --- Derived ---
    @property
    def output_path(self) -> Path:
        p = _ROOT / self.report_output_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve_device(self) -> str:
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# Module-level singleton
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# --- Constants ---

YOLO_MODELS = {
    "YOLO26n": "yolo26n.pt",
    "YOLO26m": "yolo26m.pt",
    "YOLO26l": "yolo26l.pt",
}

MODE_DEFAULT_MODELS = {
    "live": "YOLO26n",   # lowest latency for live stream
    "image": "YOLO26l",  # highest quality for still images
    "video": "YOLO26m",  # balanced speed/quality for video
}

# Per-mode resize limits (max pixel dimension before YOLO inference).
# Smaller = faster inference; larger = more detail preserved.
MODE_MAX_DIM = {
    "live": 640,    # fast — minimise per-frame latency
    "video": 960,   # balanced — throughput vs quality
    "image": 1280,  # quality — full resolution for stills
}

# Per-mode tracking behaviour
MODE_USES_TRACKING = {
    "live": True,    # track() with persist for real-time continuity
    "video": True,   # track() for temporal coherence across frames
    "image": False,  # detect() only — single frame, no sequence
}

# Per-mode video display throttle: render to UI every N processed frames
VIDEO_DISPLAY_INTERVAL = 5

PROVIDERS = ["Ollama", "OpenAI", "Gemini", "Anthropic"]
