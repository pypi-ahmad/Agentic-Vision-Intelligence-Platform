# Agentic Vision Intelligence Platform

A modular computer-vision and generative-AI application that combines **YOLO26 object detection**, **LangGraph agentic orchestration**, and **multi-provider LLM reasoning** behind a single Streamlit interface. Upload an image, process a video, or stream a live camera — the platform detects objects, tracks state across frames, extracts events, triggers intelligent LLM analysis, and lets you ask questions or generate reports about what it sees.

---

## Table of Contents

- [Key Capabilities](#key-capabilities)
- [Demo Workflow](#demo-workflow)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Input Modes](#input-modes)
- [YOLO26 Variants & Override Logic](#yolo26-variants--override-logic)
- [LLM Providers](#llm-providers)
- [API Key Handling](#api-key-handling)
- [Model Selection](#model-selection)
- [LangGraph Orchestration](#langgraph-orchestration)
- [Session Export](#session-export)
- [Architecture Overview](#architecture-overview)
- [CI / Quality Gates](#ci--quality-gates)
- [Troubleshooting](#troubleshooting)
- [Limitations & Known Trade-offs](#limitations--known-trade-offs)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Key Capabilities

| Area | What It Does |
|---|---|
| **Object Detection** | YOLO26 in three variants (nano / medium / large) with mode-specific defaults and user override |
| **Multi-Provider LLM** | Ollama (local-first), OpenAI, Google Gemini, Anthropic — dynamically lists models, stores keys per-provider |
| **Agentic Orchestration** | Three LangGraph state-graph workflows: Perception, Q&A, and Report generation with conditional routing |
| **Three Input Modes** | Single image upload, batch video processing with tracking, and real-time camera feed |
| **Smart LLM Gating** | LLM is invoked only on warning/alert events or threshold crossings — never on every frame |
| **Event Intelligence** | Automatic extraction of scene events (object appeared/left, count change, crowding, new tracks) with cooldown-based deduplication |
| **Alert System** | Severity classification (info / warning / alert) with in-app acknowledgement |
| **Scene Memory** | Rolling-window tracked-object state with time-based eviction |
| **Session Export** | Markdown reports, JSON event logs, session summaries, config snapshots, chat history, and text timelines |

---

## Demo Workflow

```
1. Launch the app          → streamlit run src/ui/streamlit_app.py
2. Select input mode       → Image / Video / Live
3. Choose a YOLO variant   → YOLO26n (fast) / YOLO26m (balanced) / YOLO26l (accurate)
4. Pick an LLM provider    → Ollama, OpenAI, Gemini, or Anthropic
5. Click Initialise        → Pipeline boots CV + optional LLM
6. Upload or start capture → Detection runs, events are extracted, alerts fire
7. Ask questions           → "What objects are in the scene?" via the Q&A tab
8. Generate a report       → Session summary in Markdown
9. Export the session       → JSON + Markdown artefacts saved to disk
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | Required |
| **Ollama** | Optional — for local LLM inference without API keys. Install from [ollama.com](https://ollama.com/) |
| **Webcam** | Optional — only needed for live camera mode |
| **GPU (CUDA)** | Optional — YOLO26 auto-selects GPU when available, falls back to CPU |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/pypi-ahmad/Agentic-Vision-Intelligence-Platform.git
cd "Agentic-Vision-Intelligence-Platform"

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

YOLO26 model weights (`yolo26n.pt`, `yolo26m.pt`, `yolo26l.pt`) are **auto-downloaded** by Ultralytics on first use. No manual download is required.

---

## Configuration

Copy the environment template and fill in values as needed:

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | If using OpenAI | — | OpenAI API key |
| `GEMINI_API_KEY` | If using Gemini | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | — | Anthropic API key |
| `YOLO_CONFIDENCE` | No | `0.35` | Default detection confidence threshold (0.0–1.0) |
| `CAMERA_INDEX` | No | `0` | OpenCV camera device index |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | No | `1280` / `720` | Requested camera resolution |
| `FRAME_SAMPLE_RATE` | No | `5` | Process every Nth frame in video/live modes |
| `MEMORY_WINDOW_SECONDS` | No | `300` | Rolling window for tracked-object retention |
| `EVENT_COOLDOWN_SECONDS` | No | `10` | Minimum interval between duplicate events |
| `LLM_TRIGGER_THRESHOLD` | No | `3` | Event-count bucket size that triggers an LLM summary |

All configuration is managed through **Pydantic Settings** (`config/__init__.py`) with `.env` file support. API keys can also be entered directly in the Streamlit sidebar — they remain in the current session only and are never persisted to disk.

---

## Running the Application

```bash
streamlit run src/ui/streamlit_app.py
```

**Recommended local-first workflow:**

1. Start with **Ollama** as the provider (no API key required).
2. Click **Fetch Models** or enter a model name manually.
3. Click **Initialise Pipeline** — the app works in CV-only mode if no LLM is configured.
4. Switch to a cloud provider when you need stronger reasoning or multimodal capabilities.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov=config --cov-report=term-missing
```

The test suite contains **149 unit tests** covering config, input sources, vision models, memory, orchestration, providers, reporting, session store, and utilities. All tests are hermetic — mocked I/O, no network calls, no filesystem side-effects.

---

## Project Structure

```
.
├── config/
│   └── __init__.py              # Pydantic Settings, YOLO_MODELS, MODE_* constants, PROVIDERS
├── src/
│   ├── input/
│   │   ├── base.py              # FramePacket dataclass, InputSource ABC
│   │   ├── camera.py            # CameraSource — live OpenCV capture
│   │   ├── image.py             # ImageSource (file), ImageArraySource (in-memory)
│   │   └── video.py             # VideoSource — video file frame extraction
│   ├── vision/
│   │   ├── models.py            # YOLO26 model loading with variant-level caching
│   │   ├── detector.py          # VisionDetector — detect() and track() with Detection/FrameResult
│   │   └── events.py            # EventExtractor — cooldown-gated SceneEvent extraction
│   ├── memory/
│   │   ├── scene_state.py       # SceneState — rolling-window tracked-object state
│   │   ├── event_timeline.py    # EventTimeline — queryable event history with severity filtering
│   │   └── session_store.py     # SessionStore — JSON/text persistence for session artefacts
│   ├── providers/
│   │   ├── base.py              # LLMProvider ABC, ProviderError hierarchy
│   │   ├── ollama_provider.py   # OllamaProvider — local HTTP API
│   │   ├── openai_provider.py   # OpenAIProvider — GPT models via openai SDK
│   │   ├── gemini_provider.py   # GeminiProvider — Google Gemini via google-genai SDK
│   │   ├── anthropic_provider.py # AnthropicProvider — Claude models via anthropic SDK
│   │   └── __init__.py          # get_provider() factory
│   ├── reasoning/
│   │   ├── prompts.py           # System prompts and task-specific prompt templates
│   │   └── reasoner.py          # Reasoner — task-method wrapper around any LLMProvider
│   ├── orchestration/
│   │   ├── state.py             # PipelineState TypedDict (25+ fields, append reducer for debug trace)
│   │   ├── nodes.py             # 11 node functions, singleton management, session reset
│   │   ├── graph.py             # 3 graph builders, 4 routing functions, compiled workflow cache
│   │   └── __init__.py          # Public API — process_frame(), ask_question(), generate_report()
│   ├── reporting/
│   │   ├── alerts.py            # AlertManager — severity-based ingestion and acknowledgement
│   │   └── exporter.py          # SessionExporter — structured session export to disk
│   ├── utils/
│   │   └── frame_utils.py       # BGR/RGB conversion, PIL/numpy, proportional resize
│   └── ui/
│       └── streamlit_app.py     # Single Streamlit entry point — sidebar, workspace, result tabs
├── tests/                       # 149 hermetic unit tests (pytest)
├── data/                        # Sample input data (gitignored weights)
├── output/                      # Session export output directory
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
├── ARCHITECTURE.md              # Detailed architecture documentation
└── README.md
```

---

## Input Modes

### Image Mode

- **Default model:** YOLO26l (highest accuracy)
- **Pipeline:** Single-frame `detect()` — no tracking
- **Max resize:** 1280px (longest edge)
- **Caching:** Detection results are cached by filename + size + model + confidence to avoid re-running YOLO on Streamlit reruns
- **Layout:** Side-by-side original and detection columns, followed by result tabs

### Video Mode

- **Default model:** YOLO26m (balanced)
- **Pipeline:** Frame-by-frame `track()` with persistence — tracking IDs carry across frames
- **Max resize:** 960px (longest edge)
- **Sampling:** Processes every Nth frame (configurable via `FRAME_SAMPLE_RATE`)
- **Display throttling:** UI refreshes every 5 processed frames (`VIDEO_DISPLAY_INTERVAL`) to keep Streamlit responsive
- **Cleanup:** Temporary video files are removed after processing

### Live Camera Mode

- **Default model:** YOLO26n (lowest latency)
- **Pipeline:** `track()` with persistence, frames resized to 640px
- **Frame skipping:** Reads and discards stale buffered frames before processing the latest one
- **LLM suppression:** Periodic auto-summaries are suppressed in live mode to reduce latency; use the Q&A tab for on-demand analysis
- **Cycle:** Each processed frame triggers `st.rerun()` for the next capture cycle

---

## YOLO26 Variants & Override Logic

| Variant | Weight File | Default For | Max Resize | Tracking | Optimised For |
|---|---|---|---|---|---|
| **YOLO26n** | `yolo26n.pt` | Live | 640px | Yes | Low latency, real-time streaming |
| **YOLO26m** | `yolo26m.pt` | Video | 960px | Yes | Balanced throughput and accuracy |
| **YOLO26l** | `yolo26l.pt` | Image | 1280px | No | Maximum detection quality |

**Override logic:** When the user switches input mode, the YOLO variant auto-selects the recommended default. The user can then manually override the variant from the sidebar dropdown. Overrides are respected — the app displays a note in the status bar when a non-default variant is selected. Changing the variant resets the detector and requires re-initialisation.

Weights are auto-downloaded by Ultralytics on first use and cached locally. They are gitignored (`*.pt`).

---

## LLM Providers

All four providers implement the same `LLMProvider` abstract interface:

| Provider | SDK | Vision Support | Key Required |
|---|---|---|---|
| **Ollama** | `httpx` (HTTP API) | Yes (base64 in payload) | No — local inference |
| **OpenAI** | `openai` | Yes (`image_url` content blocks) | Yes |
| **Gemini** | `google-genai` | Yes (PIL image objects) | Yes |
| **Anthropic** | `anthropic` | Yes (base64 `image` content blocks) | Yes |

Each adapter maps SDK-specific exceptions to the typed `ProviderError` hierarchy:
- `ProviderAuthError` — invalid or missing credentials
- `ProviderConnectionError` — network unreachable or timeout
- `ProviderRateLimitError` — provider-side rate limiting

If model listing fails for any reason, the UI degrades gracefully to a manual text input instead of blocking the workflow.

---

## API Key Handling

API keys can be provided in two ways:

1. **Environment variables** — set `OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY` in `.env`. These are loaded once at startup via Pydantic Settings.
2. **Sidebar input** — enter keys directly in the Streamlit sidebar. Keys are stored in `st.session_state` (browser-tab-scoped memory) and are **never written to disk**.

Ollama requires no API key. When switching between cloud providers, previously entered keys are preserved in session state so you do not need to re-enter them.

The factory function `get_provider()` raises `ProviderAuthError` immediately if a cloud provider is requested without a key, before any network call is made.

---

## Model Selection

Model selection follows a two-tier strategy:

1. **Dynamic fetch** — Click **Fetch Models** in the sidebar. The app calls `list_models()` on the active provider and populates a dropdown. The previously selected model is preserved if it still exists in the new list.
2. **Manual fallback** — If fetching fails (network error, auth error, empty response), a free-text input appears so you can type a model name directly (e.g. `llama3`, `gpt-4o`, `gemini-2.0-flash`).

Provider-specific model filtering:
- **OpenAI** filters out non-chat models (embeddings, TTS, Whisper, DALL-E, moderation, legacy completions).
- **Gemini** filters for models containing `"gemini"` in the name.
- **Anthropic** returns all models from the SDK's list endpoint.
- **Ollama** returns all locally pulled models.

---

## LangGraph Orchestration

The platform uses [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph` to define three declarative workflows over a shared `PipelineState` TypedDict with 25+ typed fields.

### Perception Graph (9 nodes)

Runs on every frame via `process_frame()`:

```
ingest → run_cv → extract_events → update_memory → detect_change
           ↓                                            ↓
       [finalize]                              ┌────────┴────────┐
       (on error)                         create_alert      decide_reasoning
                                               ↓                 ↓
                                        decide_reasoning    ┌────┴────┐
                                               ↓         call_llm  finalize
                                          ┌────┴────┐       ↓
                                       call_llm  finalize  finalize
                                           ↓
                                        finalize
```

**Conditional routing:**
- `route_after_ingest` — skips to `finalize` on missing frame
- `route_after_change` — routes to `create_alert` when warning/alert-severity events are present
- `route_after_reasoning` — routes to `call_llm` only when `llm_needed` is set (threshold crossing, anomaly, or explicit task)

**LLM gating in live mode:** Periodic bucket-based summaries are suppressed to preserve latency. LLM is still invoked for warnings, alerts, and explicit user-triggered tasks.

### Q&A Graph (4 nodes)

Runs on user questions via `ask_question()`:

```
qa_cv → [update_memory →] call_llm → finalize
```

If a frame and detector are available, `qa_cv` runs detection first so the LLM answer is grounded in fresh CV data. Otherwise, it uses the existing scene context.

### Report Graph (3 nodes)

Runs on demand via `generate_report()`:

```
gather_report_context → call_llm → finalize
```

Assembles session metadata, scene state, event history, and object statistics, then invokes the LLM with the report prompt template.

### Debug Tracing

Every node appends its name to `_node_trace` (an `Annotated[list, operator.add]` field), providing a full execution path log for each workflow invocation.

---

## Session Export

Clicking **Export Session** in the Export tab writes a timestamped session folder under `output/` containing:

| File | Format | Contents |
|---|---|---|
| `events.json` | JSON | Full event history |
| `alerts.json` | JSON | All alerts with acknowledgement status |
| `summary.json` | JSON | Scene state snapshot |
| `report.md` | Markdown | LLM-generated session report (if generated) |
| `config_snapshot.json` | JSON | Mode, model variant, confidence, provider, model name, session ID, duration |
| `chat_history.json` | JSON | Full Q&A conversation log |
| `event_timeline.txt` | Text | Human-readable `[HH:MM:SS] [SEVERITY] description` timeline |

---

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full technical deep-dive, including layer diagrams, data flow, module interfaces, and design decisions.

```
┌─────────────────────────────────────────────────┐
│                  Streamlit UI                    │
│           src/ui/streamlit_app.py                │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│           Orchestration (LangGraph)              │
│   Perception Graph │ QA Graph │ Report Graph     │
└───┬──────┬─────────┬─────────┬──────────────────┘
    │      │         │         │
┌───▼──┐ ┌─▼──────┐ ┌▼───────┐ ┌▼──────────┐
│Vision│ │Memory  │ │Reason- │ │Reporting  │
│      │ │        │ │ing     │ │           │
└──┬───┘ └────────┘ └──┬─────┘ └───────────┘
   │                   │
┌──▼───────────────────▼─────────────────────┐
│              Provider Layer                 │
│   Ollama │ OpenAI │ Gemini │ Anthropic      │
└────────────────────────────────────────────┘
```

---

## CI / Quality Gates

```bash
# Unit tests (149 tests, hermetic, ~4s)
pytest tests/ -v

# Tests with coverage
pytest tests/ -v --cov=src --cov=config --cov-report=term-missing

# Lint
ruff check .

# Type check
mypy src/ --ignore-missing-imports
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` on launch | Virtual environment not activated or dependencies missing | Activate `.venv` and run `pip install -r requirements.txt` |
| "Cannot open camera" in live mode | No webcam connected, or `CAMERA_INDEX` wrong | Check device connection; try `CAMERA_INDEX=1` in `.env` |
| Fetch Models returns empty for Ollama | No models pulled locally | Run `ollama pull llama3` (or any model) in a terminal first |
| "Missing or invalid API key" | Cloud provider key not set | Enter the key in the sidebar or add it to `.env` |
| YOLO weights download stalls | Network issue during first-time Ultralytics download | Retry, or manually download `.pt` files to the working directory |
| Streamlit reruns feel slow in video mode | Large video with low sample rate | Increase `FRAME_SAMPLE_RATE` in `.env` |
| LLM not responding in live mode | Periodic summaries are suppressed by design | Use the Q&A tab for on-demand analysis during live mode |
| "Rate limited" warning | Provider-side throttling | Wait a moment and retry, or switch to a different provider/model |
| Export directory not found | `output/` folder permissions | Ensure the process has write access to the project root |

---

## Limitations & Known Trade-offs

- **Single-user Streamlit deployment.** The app runs as a single Streamlit process with module-level singletons. It is not designed for concurrent multi-user access. Each browser tab shares the same backend state.
- **No REST API.** There is no FastAPI or Flask layer. The app is interactive-only via the Streamlit UI.
- **Blocking video processing.** Video mode processes frames synchronously in a loop. Long videos block the UI until complete. There is no background worker or task queue.
- **Live mode uses rerun polling.** The camera feed works via `st.rerun()` cycles, not WebSocket streaming. Frame rate is limited by Streamlit's rerun overhead.
- **No model fine-tuning.** YOLO26 models are used with pretrained COCO weights. Custom object classes require retraining outside this application.
- **LLM latency.** Cloud LLM calls add 1–5 seconds per invocation. The smart gating system mitigates this by only calling the LLM when meaningful events occur.
- **Provider vision varies.** Each LLM provider handles image input differently (base64, PIL, URL). Some models within a provider may not support vision at all.
- **No authentication.** The app has no user login or access control. It is designed for local or trusted-network use.
- **Event extraction heuristics.** The `EventExtractor` uses fixed thresholds (e.g. ≥10 for crowding, ≥50% for count change). These are not configurable without code changes.

---

## Future Improvements

- **Multi-camera support** — process multiple simultaneous camera feeds with per-source state
- **Async video processing** — background worker for video mode to keep the UI responsive
- **WebSocket camera streaming** — replace rerun-based polling with a real-time streaming protocol
- **Configurable event thresholds** — expose crowding limit, count-change percentage, and other heuristics as settings
- **Custom YOLO model support** — allow users to upload or point to custom-trained `.pt` weights
- **Persistent session database** — store session history in SQLite or PostgreSQL instead of flat files
- **User authentication** — add login and role-based access for multi-user deployments
- **REST API layer** — optionally expose the orchestration layer as a REST/gRPC API for programmatic access
- **Batch image processing** — support uploading multiple images for bulk analysis
- **Dashboard mode** — historical analytics view across multiple sessions with trend charts

---

## License

MIT
