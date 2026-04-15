# Architecture

## Overview

The Agentic Vision Intelligence Platform is a modular, layered application that combines computer vision (YOLO26), generative AI (multiple LLM providers), and agentic orchestration (LangGraph) behind a Streamlit UI.

## Layer Diagram

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│           src/ui/streamlit_app.py                │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│              Orchestration (LangGraph)           │
│          src/orchestration/__init__.py           │
│   Perception Graph │ QA Graph │ Report Graph     │
└───┬──────┬─────────┬─────────┬──────────────────┘
    │      │         │         │
┌───▼──┐ ┌─▼──────┐ ┌▼───────┐ ┌▼──────────┐
│Vision│ │Memory  │ │Reason  │ │Reporting  │
│Layer │ │Layer   │ │Layer   │ │Layer      │
└──┬───┘ └──┬─────┘ └──┬─────┘ └──┬────────┘
   │        │          │          │
┌──▼────────▼──────────▼──────────▼──────────┐
│              Provider Layer                 │
│   Ollama │ OpenAI │ Gemini │ Anthropic      │
└─────────────────────────────────────────────┘
```

## Module Details

### `config/`
Central configuration via Pydantic Settings. Defines `YOLO_MODELS`, `MODE_DEFAULT_MODELS`, and `PROVIDERS` constants. Loads environment variables from `.env`.

### `src/input/`
Abstract `InputSource` base class with concrete implementations:
- `CameraSource` — OpenCV webcam capture
- `ImageSource` — single image file loading
- `ImageArraySource` — uploaded/in-memory image frames
- `VideoSource` — video file frame extraction

The Streamlit UI uses these input abstractions directly instead of bypassing them with ad hoc OpenCV logic.

### `src/vision/`
- **`models.py`** — YOLO26 model loading with caching
- **`detector.py`** — `VisionDetector` wraps any YOLO26 variant, supports detect() and track()
- **`events.py`** — `EventExtractor` compares consecutive frames and emits `SceneEvent`s with cooldown

### `src/memory/`
- **`scene_state.py`** — rolling window of tracked objects and counts
- **`event_timeline.py`** — queryable event history with severity filtering
- **`session_store.py`** — JSON/text file persistence for export-side artefacts such as config snapshots, chat history, and text timelines

### `src/providers/`
Abstract `LLMProvider` interface with four concrete adapters:
- `OllamaProvider` — local inference via HTTP API
- `OpenAIProvider` — GPT models via openai SDK
- `GeminiProvider` — Google Gemini via google-genai SDK
- `AnthropicProvider` — Claude models via anthropic SDK

Factory function `get_provider(name, api_key, ollama_url)` in `__init__.py`.

### `src/reasoning/`
- **`prompts.py`** — system prompts and task templates
- **`reasoner.py`** — `Reasoner` class wrapping any provider with task methods

### `src/orchestration/`
LangGraph-based agentic workflows:
- **Perception Graph**: run_cv → extract_events → update_memory → create_alert → decide_llm → [call_llm | END]
  - `decide_llm` uses threshold-crossing logic to avoid unnecessary LLM calls
  - LLM is only invoked when: warning/alert events detected, event count crosses threshold, or explicit task requested
- **QA Graph**: maybe_cv (skips if no frame) → update_memory → call_llm
  - Allows Q&A about existing context without requiring a new frame
- **Report Graph**: call_llm (uses accumulated scene state and event history)

High-level functions: `process_frame()`, `ask_question()`, `generate_report()`
Helper: `_dets_from_state()` / `_frame_result_from_state()` for DRY Detection reconstruction

### `src/reporting/`
- **`alerts.py`** — `AlertManager` with severity-based ingestion and acknowledgement
- **`exporter.py`** — `SessionExporter` saves reports, events, alerts, summaries to disk

`SessionExporter` and `SessionStore` are used together in the UI export flow so structured JSON and human-readable artefacts land in the same session folder.

### `src/ui/`
Single Streamlit application. Sidebar controls: mode, CV model (with mode-default recommendation), confidence, LLM provider, per-provider API key storage, model selection (dynamic fetch or manual entry), and clear CV-vs-LLM readiness feedback. Main workspace: image/video/camera views with tabs for summary, events, alerts, Q&A, report, and export. Image detection results are cached to avoid re-running YOLO on Streamlit reruns. Live mode uses `CameraSource` with a batch-and-rerun pattern instead of a blocking loop.

## Data Flow

1. User uploads image/video or starts camera → frame extracted through `InputSource` implementations
2. Frame enters LangGraph perception workflow
3. YOLO26 runs detection/tracking → `FrameResult`
4. Events extracted via `EventExtractor` → `SceneEvent`s
5. Memory updated (scene state + event timeline)
6. Alerts created for warning/alert-severity events
7. LLM optionally invoked for description, summary, or anomaly reasoning
8. Results rendered in Streamlit tabs
9. User can ask questions (QA graph) or generate reports (Report graph)
10. Session data exportable to disk as both machine-readable JSON and operator-friendly text/markdown

## Design Decisions

- **Streamlit-only UI** — no REST API layer; the app is interactive and stateful
- **Mode-specific CV defaults** — YOLO26n for live (latency), YOLO26m for video (balanced), YOLO26l for image (accuracy); user can override
- **Provider abstraction** — all LLM providers share the same interface; switching is seamless
- **Per-provider API key storage** — switching providers preserves previously entered keys
- **Smart LLM gating** — threshold-crossing logic prevents LLM spam on every frame; only triggers on warning events, threshold crossings, or explicit user requests
- **LangGraph orchestration** — declarative node graphs with conditional routing for perception, QA, and report generation
- **Cooldown-based event deduplication** — prevents flooding the timeline with repetitive events
- **Detection caching** — image mode caches YOLO results to avoid re-inference on Streamlit reruns
- **Shared input abstractions** — image, video, and live camera all flow through the same input layer used by the rest of the architecture
- **Graceful degradation** — all providers handle fetch failures silently; manual model entry available as fallback
