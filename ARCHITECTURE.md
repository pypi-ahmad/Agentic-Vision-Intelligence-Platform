# Architecture

> Technical reference for the Agentic Vision Intelligence Platform.
> For quick-start and usage, see [README.md](README.md).

---

## Table of Contents

- [System Overview](#system-overview)
- [Layer Diagram](#layer-diagram)
- [Data Flow](#data-flow)
- [Module Reference](#module-reference)
  - [config/](#config)
  - [src/input/](#srcinput)
  - [src/vision/](#srcvision)
  - [src/memory/](#srcmemory)
  - [src/providers/](#srcproviders)
  - [src/reasoning/](#srcreasoning)
  - [src/orchestration/](#srcorchestration)
  - [src/reporting/](#srcreporting)
  - [src/utils/](#srcutils)
  - [src/ui/](#srcui)
- [LangGraph Workflows In Depth](#langgraph-workflows-in-depth)
- [Singleton Lifecycle](#singleton-lifecycle)
- [Error Handling Strategy](#error-handling-strategy)
- [Design Decisions](#design-decisions)

---

## System Overview

The platform is a layered, modular Python application. Each layer communicates through well-defined interfaces:

1. **UI Layer** — Streamlit handles all user interaction, session state, and rendering.
2. **Orchestration Layer** — LangGraph `StateGraph` workflows coordinate CV, memory, events, and LLM calls through declarative node graphs with conditional routing.
3. **Domain Layers** — Vision (YOLO26), Memory (scene state + event timeline), Reasoning (LLM task dispatch), and Reporting (alerts + export) each operate independently behind clean interfaces.
4. **Provider Layer** — Abstract `LLMProvider` interface with four concrete adapters (Ollama, OpenAI, Gemini, Anthropic), each mapping SDK-specific exceptions to a shared error hierarchy.

---

## Layer Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        Streamlit UI                          │
│                   src/ui/streamlit_app.py                     │
│  Sidebar: mode, model, provider, keys, init, reset           │
│  Workspace: image/video/live rendering, 6-tab result display  │
└────────────────────────┬─────────────────────────────────────┘
                         │  process_frame()  ask_question()  generate_report()
┌────────────────────────▼─────────────────────────────────────┐
│                  Orchestration (LangGraph)                    │
│                 src/orchestration/                            │
│   ┌──────────────┐  ┌──────────┐  ┌──────────────┐          │
│   │  Perception   │  │   Q&A    │  │   Report     │          │
│   │  Graph (9)    │  │ Graph(4) │  │  Graph (3)   │          │
│   └──────┬───────┘  └────┬─────┘  └──────┬───────┘          │
│          │               │               │                    │
│   PipelineState (TypedDict, 25+ fields, append reducer)      │
└──┬───────┼───────┬───────┼───────┬───────┼───────────────────┘
   │       │       │       │       │       │
┌──▼───┐ ┌─▼────┐ ┌▼──────┐ ┌─────▼┐ ┌───▼──────┐
│Vision│ │Memory│ │Events │ │Reason│ │Reporting │
│      │ │      │ │       │ │-ing  │ │          │
└──┬───┘ └──────┘ └───────┘ └──┬───┘ └──────────┘
   │                           │
┌──▼───────────────────────────▼──────────────────┐
│               Provider Layer                     │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐       │
│  │Ollama│ │OpenAI│ │Gemini│ │Anthropic │       │
│  └──────┘ └──────┘ └──────┘ └──────────┘       │
│  LLMProvider ABC → ProviderError hierarchy       │
└──────────────────────────────────────────────────┘
```

---

## Data Flow

```
User Action
    │
    ▼
1.  Frame acquired via InputSource (CameraSource / ImageArraySource / VideoSource)
    │
    ▼
2.  Frame enters LangGraph perception workflow as PipelineState
    │
    ├── node_ingest         → validate frame, set error flag if missing
    ├── node_run_cv         → YOLO26 detect() or track() based on MODE_USES_TRACKING
    ├── node_extract_events → compare with previous state, emit SceneEvents with cooldown
    ├── node_update_memory  → update SceneState (tracked objects, counts, rolling window)
    ├── node_detect_change  → assess max severity of new events
    ├── node_create_alert   → promote warning/alert events to Alert objects  (conditional)
    ├── node_decide_reasoning → gate LLM: threshold crossing, anomaly, or explicit task
    ├── node_call_llm       → dispatch to Reasoner method (describe/summarize/anomaly/qa/report)
    └── node_finalize       → log execution trace
    │
    ▼
3.  Results rendered in Streamlit tabs:
    Summary (metrics + chart) │ Events │ Alerts │ Q&A │ Report │ Export
    │
    ▼
4.  User interactions:
    • Q&A tab  → ask_question() invokes Q&A graph
    • Report   → generate_report() invokes Report graph
    • Export   → SessionExporter writes JSON/Markdown/text to output/
```

---

## Module Reference

### `config/`

**File:** `config/__init__.py`

Central configuration via Pydantic `BaseSettings`. Loads from environment variables and `.env`.

**`Settings` class fields:**

| Field | Type | Default | Constraint |
|---|---|---|---|
| `ollama_base_url` | `str` | `http://localhost:11434` | — |
| `openai_api_key` | `str` | `""` | — |
| `gemini_api_key` | `str` | `""` | — |
| `anthropic_api_key` | `str` | `""` | — |
| `yolo_confidence` | `float` | `0.35` | 0.0–1.0 |
| `camera_index` | `int` | `0` | — |
| `camera_width` | `int` | `1280` | — |
| `camera_height` | `int` | `720` | — |
| `frame_sample_rate` | `int` | `5` | ≥ 1 |
| `memory_window_seconds` | `int` | `300` | ≥ 10 |
| `event_cooldown_seconds` | `int` | `10` | ≥ 1 |
| `llm_trigger_threshold` | `int` | `3` | ≥ 1 |
| `report_output_dir` | `str` | `"output"` | — |

**`resolve_device()`** returns `"0"` if CUDA is available, `"cpu"` otherwise.

**Module-level constants:**

| Constant | Value | Purpose |
|---|---|---|
| `YOLO_MODELS` | `{"YOLO26n": "yolo26n.pt", "YOLO26m": "yolo26m.pt", "YOLO26l": "yolo26l.pt"}` | Variant-to-weight mapping |
| `MODE_DEFAULT_MODELS` | `{"live": "YOLO26n", "image": "YOLO26l", "video": "YOLO26m"}` | Mode-specific defaults |
| `MODE_MAX_DIM` | `{"live": 640, "video": 960, "image": 1280}` | Max resize per mode |
| `MODE_USES_TRACKING` | `{"live": True, "video": True, "image": False}` | `track()` vs `detect()` |
| `VIDEO_DISPLAY_INTERVAL` | `5` | UI refresh throttle for video mode |
| `PROVIDERS` | `["Ollama", "OpenAI", "Gemini", "Anthropic"]` | Supported LLM providers |

---

### `src/input/`

Abstract input source layer. All frame acquisition flows through these classes.

| Class | File | Source Type | `is_live` | Key Behaviour |
|---|---|---|---|---|
| `InputSource` | `base.py` | ABC | `False` | Defines `open()`, `read()`, `close()`, `frames()` iterator |
| `FramePacket` | `base.py` | Dataclass | — | `frame`, `timestamp`, `frame_index`, `source_id`, `source_type`, `metadata` |
| `CameraSource` | `camera.py` | `"camera"` | `True` | OpenCV `VideoCapture`, configurable resolution from `Settings` |
| `ImageSource` | `image.py` | `"image"` | `False` | Reads a single image file via `cv2.imread`, validates existence |
| `ImageArraySource` | `image.py` | `"image"` | `False` | Wraps an in-memory numpy array (used for Streamlit uploads) |
| `VideoSource` | `video.py` | `"video"` | `False` | OpenCV `VideoCapture` with `fps`, `total_frames`, `duration_seconds` properties |

---

### `src/vision/`

Computer vision detection, tracking, and event extraction.

**`models.py`** — `load_yolo(variant)` loads and caches YOLO26 models by variant name. Models are cached at the module level so repeated calls return the same instance.

**`detector.py`** — `VisionDetector` wraps any YOLO26 variant:

| Method | Pipeline | Returns |
|---|---|---|
| `detect(frame)` | Single-frame prediction, no tracking | `FrameResult` |
| `track(frame, persist=True)` | Multi-frame tracking with persistent IDs | `FrameResult` |

`Detection` dataclass: `class_id`, `class_name`, `confidence`, `bbox` (x1,y1,x2,y2), optional `track_id`. Computed properties: `center`, `area`.

`FrameResult` dataclass: `frame_index`, `detections: list[Detection]`, `object_counts: dict[str,int]`, `annotated_frame: np.ndarray | None`. Computed property: `summary_line` (e.g. `"5 person, 2 car"`).

**`events.py`** — `EventExtractor` compares consecutive frames and emits `SceneEvent` objects:

| Event Type | Trigger |
|---|---|
| `object_appeared` | New class name appears in the scene |
| `object_left` | Previously present class disappears |
| `count_change` | Object count changes by ≥ 2 or ≥ 50% |
| `crowding` | Any single class count reaches ≥ 10 |
| `new_tracks` | ≥ 3 new tracked objects appear in one frame |

All events are subject to per-key cooldown (`event_cooldown_seconds`). Severity is `"info"` for appearances/departures, `"warning"` for count changes and crowding, `"alert"` for new-track surges.

---

### `src/memory/`

Scene state management and persistence.

**`SceneState`** (`scene_state.py`):
- Maintains a dict of `TrackedObject` instances keyed by `track_id`
- Rolling window: objects not seen within `memory_window_seconds` are evicted on each `update()` call
- `active_objects` — objects seen within the last 5 seconds
- `get_summary()` — structured dict with timestamp, counts, active tracked count, class names
- `get_description()` — human-readable scene description string

**`EventTimeline`** (`event_timeline.py`):
- Append-only list of `SceneEvent` objects, capped at `max_events` (default: 1000)
- Queryable: `recent(n)`, `by_severity(severity)`, `warnings_and_alerts()`
- `to_text()` — formatted `[HH:MM:SS] [SEVERITY] description` for each event
- `get_summary()` — type counts, severity counts, 5 most recent events

**`SessionStore`** (`session_store.py`):
- Writes JSON and text files to a session-scoped directory under `output/`
- Used by the export flow for config snapshots, chat history, and event timelines

---

### `src/providers/`

Four LLM provider adapters behind a shared abstract interface.

**Error hierarchy** (`base.py`):
```
ProviderError (base)
├── ProviderAuthError       — invalid/missing credentials
├── ProviderConnectionError — network failure or timeout
└── ProviderRateLimitError  — provider-side rate limiting
```

**`LLMProvider` ABC** (`base.py`):
- `list_models() -> list[str]` — fetch available models
- `generate(prompt, *, model, images, system) -> str` — text + optional vision inference
- `is_available() -> bool` — tries `list_models()`, returns `False` on any exception
- `_encode_image_b64(image, quality=85) -> str` — static JPEG base64 helper

**Adapter details:**

| Adapter | SDK | Vision Handling | Model Filtering |
|---|---|---|---|
| `OllamaProvider` | `httpx` (REST) | Base64 in JSON payload | All local models |
| `OpenAIProvider` | `openai` | `image_url` content blocks | Filters out embeddings, TTS, DALL-E, legacy |
| `GeminiProvider` | `google-genai` | PIL image objects | Models containing `"gemini"` |
| `AnthropicProvider` | `anthropic` | Base64 `image` content blocks | All from SDK list endpoint |

**Factory** (`__init__.py`): `get_provider(name, *, api_key, ollama_url)` dispatches by name, raises `ProviderAuthError` pre-emptively for cloud providers without keys.

---

### `src/reasoning/`

LLM task dispatch with structured prompt templates.

**`prompts.py`** — String constants with `str.format()` placeholders:

| Template | Task | Format Keys |
|---|---|---|
| `SCENE_DESCRIPTION` | Describe scene from CV data | `detection_summary`, `object_counts`, `tracked_objects` |
| `EVENT_SUMMARY` | Summarise recent events | `events_text`, `scene_state` |
| `QUESTION_ANSWER` | Answer user question | `scene_description`, `recent_events`, `detection_data`, `question` |
| `INCIDENT_REPORT` | Generate session report | `session_id`, `duration`, `total_frames`, `scene_summary`, `events_text`, `object_stats` |
| `ANOMALY_REASONING` | Analyse anomalies | `detection_data`, `events_text`, `scene_description` |
| `ALERT_EXPLANATION` | Explain a specific alert | `alert_type`, `severity`, `description`, `scene_context`, `recent_events` |

System prompts: `SYSTEM_ANALYST`, `SYSTEM_REPORT`, `SYSTEM_QA`.

**`Reasoner`** (`reasoner.py`): Wraps any `LLMProvider` + model name. One method per task — each formats its template and calls `provider.generate()`.

---

### `src/orchestration/`

LangGraph `StateGraph` workflows with shared state and conditional routing.

**`PipelineState`** (`state.py`): `TypedDict` with `total=False` (all fields optional). 25+ fields grouped into: Input, CV Results, Scene Memory, Events, Change Detection, Reasoning, Q&A, Alerts, Reports, Flow Control, Debug. The `_node_trace` field uses `Annotated[list[str], operator.add]` for append-only merge semantics. Factory function `empty_state()` returns a clean initial state dict with all defaults.

**`nodes.py`**: 11 node functions (each takes `state: PipelineState` and returns a partial state dict), plus singleton management (`set_detector`, `set_reasoner_obj`, `clear_reasoner`, `get_scene_state`, `get_event_timeline`, `get_reasoner`, `reset_all`).

**`graph.py`**: Three graph builders with four routing functions:

| Graph | Nodes | Entry Point | Conditional Edges |
|---|---|---|---|
| Perception | 9 | `node_ingest` | `route_after_ingest`, `route_after_change`, `route_after_reasoning` |
| Q&A | 4 | `node_qa_cv` | `route_qa_after_cv` |
| Report | 3 | `node_gather_report_context` | None (linear) |

Compiled workflows are lazily cached via `get_perception_wf()`, `get_qa_wf()`, `get_report_wf()`.

**Public API** (`__init__.py`): High-level functions: `process_frame()`, `ask_question()`, `generate_report()`, `reset_session()`, `set_detector()`, `set_reasoner()`, `clear_reasoner()`, `get_scene_state()`, `get_event_timeline()`, `get_reasoner()`. The module also re-exports all node functions, graph builders, routing functions, and state types via `__all__` for direct testing and advanced use.

---

### `src/reporting/`

Alert management and session export.

**`AlertManager`** (`alerts.py`):
- `ingest_events(events)` — creates `Alert` objects for events with severity `"warning"` or `"alert"`
- `acknowledge(alert_id)` — marks an alert as acknowledged
- Properties: `unacknowledged`, `all_alerts`

**`SessionExporter`** (`exporter.py`):
- Creates a timestamped session directory under `output/`
- Methods: `save_report()`, `save_events()`, `save_alerts()`, `save_summary()`, `save_text()`

---

### `src/utils/`

**`frame_utils.py`** — Pure functions for frame processing:

| Function | Purpose |
|---|---|
| `bgr_to_rgb(frame)` | OpenCV BGR → RGB |
| `rgb_to_bgr(frame)` | RGB → BGR |
| `pil_to_numpy(img)` | PIL Image → numpy RGB array |
| `numpy_to_pil(arr)` | Numpy → PIL Image |
| `resize_frame(frame, max_dim)` | Proportional resize if any dimension exceeds `max_dim` (uses `cv2.INTER_AREA`) |

---

### `src/ui/`

**`streamlit_app.py`** — Single Streamlit entry point.

**Sidebar sections:**
1. Input Source — radio toggle (image / video / live) with auto-default model switching
2. CV Model — YOLO26 variant dropdown with mode recommendation, confidence slider
3. LLM Provider — provider dropdown, per-provider API key input, Fetch Models button with typed error feedback, model selector with manual fallback
4. Pipeline Status — colour-coded readiness indicators (CV ✔/✗, LLM ✔/✗)
5. Session Controls — Initialise Pipeline (primary), Reset Session

**Main workspace:**
- 4-column status bar (Mode, Model, Pipeline, Session)
- Mode-specific renderer (image: side-by-side columns; video: progress bar + metrics; live: toggle + rerun loop)
- 6-tab result display: Summary, Events, Alerts, Q&A, Report, Export

**Key patterns:**
- Detection caching via `_img_cache_key` in session state (image mode)
- `st.rerun()` polling loop for live mode camera cycles
- Video display throttling via `VIDEO_DISPLAY_INTERVAL`
- Per-provider key preservation across provider switches

---

## LangGraph Workflows In Depth

### Why LangGraph

LangGraph provides:
- **Declarative node graphs** with typed state — the perception pipeline is described as nodes and edges, not imperative function chains
- **Conditional routing** — the graph dynamically decides whether to create alerts, invoke the LLM, or skip directly to finalisation
- **State merging** — `PipelineState` is a `TypedDict` where each node returns only the fields it modifies; LangGraph merges partial updates automatically
- **Append reducers** — `_node_trace` uses `Annotated[list, operator.add]` so every node's trace entry is appended, never overwritten
- **Compiled execution** — graphs are compiled once and cached, so repeated invocations have minimal overhead

### Perception Graph Detail

```
START
  │
  ▼
node_ingest ──── error? ──→ node_finalize → END
  │ ok
  ▼
node_run_cv
  │
  ▼
node_extract_events
  │
  ▼
node_update_memory
  │
  ▼
node_detect_change ──── warning/alert? ──→ node_create_alert
  │ info only                                    │
  ▼                                              ▼
node_decide_reasoning ◄──────────────── node_decide_reasoning
  │
  ├── llm_needed=True  ──→ node_call_llm → node_finalize → END
  │
  └── llm_needed=False ──→ node_finalize → END
```

### LLM Gating Logic (`node_decide_reasoning`)

The node sets `llm_needed=True` when any of these conditions is met:
1. **Explicit task** — `reasoning_task` is already set in state (e.g. `"qa"`, `"report"`)
2. **Anomaly detection** — new events include `"warning"` or `"alert"` severity → sets `reasoning_task="anomaly"`
3. **Periodic summary** — total event count crosses a bucket threshold (`llm_trigger_threshold`) → sets `reasoning_task="summarize"` — **suppressed in live mode** to avoid latency

### Q&A Graph Detail

```
START → node_qa_cv ──── cv_ran? ──→ node_update_memory → node_call_llm → node_finalize → END
                   └── no cv ──→ node_call_llm → node_finalize → END
```

If a frame and detector are available, fresh CV data is generated before answering so the response is grounded in current detections.

---

## Singleton Lifecycle

The orchestration layer manages session-scoped singletons in `nodes.py`:

| Singleton | Created By | Reset By |
|---|---|---|
| `_detector: VisionDetector` | `set_detector(variant, confidence)` | `reset_all()` |
| `_reasoner: Reasoner` | `set_reasoner_obj(provider, model)` | `clear_reasoner()` / `reset_all()` |
| `_scene_state: SceneState` | Module init | `reset_all()` |
| `_event_timeline: EventTimeline` | Module init | `reset_all()` |
| `_event_extractor: EventExtractor` | Module init | `reset_all()` |

`reset_all()` is called by the UI's Reset Session button and clears all accumulated state.

---

## Error Handling Strategy

| Layer | Strategy |
|---|---|
| **Provider** | SDK exceptions mapped to `ProviderAuthError`, `ProviderConnectionError`, `ProviderRateLimitError`. Unknown errors wrapped in base `ProviderError`. |
| **Factory** | Pre-validates API keys before constructing provider. Raises `ProviderAuthError` immediately for empty cloud keys. |
| **UI — Fetch Models** | Catches each typed error and sets `_model_fetch_error` in session state. Sidebar renders colour-coded feedback (🔑 auth, 🌐 connection, ⏳ rate limit). |
| **UI — Initialise** | Catches `ProviderAuthError` and `ProviderConnectionError` separately with specific user guidance. Falls back to CV-only mode on LLM failure. |
| **Orchestration** | `node_ingest` sets `error` flag for missing frames; `route_after_ingest` skips CV on error. Other nodes fail-safe — a node exception does not crash the graph. |
| **Session State** | Manual model entry fallback when dynamic fetch fails. Previously entered API keys persist across provider switches. |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Streamlit-only UI** | Single runtime for interactive, stateful use. No REST API complexity for a tool that is inherently visual and session-based. |
| **Mode-specific CV defaults** | YOLO26n for live (latency), YOLO26m for video (balance), YOLO26l for image (quality). User can override. |
| **Provider abstraction** | All LLM providers share `LLMProvider` interface. Switching providers is a dropdown change, not a code change. |
| **Per-provider key storage** | `st.session_state` preserves keys per provider. Switching providers and back does not lose keys. |
| **Smart LLM gating** | Threshold-crossing + anomaly-trigger logic prevents LLM calls on every frame. Live mode further suppresses periodic summaries. |
| **LangGraph over imperative chains** | Declarative graphs are easier to extend (add a node), reason about (visible edges), and debug (append trace). |
| **Cooldown-based event dedup** | Prevents flooding the timeline with repetitive events while ensuring genuine state changes are captured. |
| **Detection caching (image mode)** | `_img_cache_key` prevents re-running YOLO on Streamlit reruns triggered by unrelated state changes. |
| **Shared input abstractions** | Image, video, and live camera all produce `FramePacket` objects. The rest of the pipeline never knows the source type. |
| **Graceful degradation** | LLM fetch failures fall back to manual entry. Missing LLM does not block CV. Missing CV does not block Q&A on cached context. |
