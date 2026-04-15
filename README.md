# Agentic Vision Intelligence Platform

A production-grade computer vision + generative AI + agentic AI application built with **Streamlit**, **Ultralytics YOLO26**, **LangGraph**, and multiple LLM providers.

## Features

| Capability | Details |
|---|---|
| **Computer Vision** | YOLO26 in three variants — nano (n), medium (m), large (l) — with automatic mode-specific defaults and user override |
| **LLM Providers** | Ollama (local), OpenAI, Google Gemini, Anthropic — with dynamic model listing, per-provider API key storage, and manual model entry fallback |
| **Agentic Orchestration** | LangGraph workflows for perception (CV → events → memory → alert → conditional LLM), Q&A (optional CV → LLM), and reporting (LLM) |
| **Input Sources** | Image upload, video upload, live camera feed through shared input abstractions |
| **Smart LLM Gating** | LLM calls only on warning/alert events or threshold crossings — not every frame |
| **Event Intelligence** | Automatic event extraction with cooldown-based deduplication |
| **Alert System** | Warning/alert severity classification with acknowledgement UI |
| **Session Export** | Markdown reports, JSON event logs, session summaries, config snapshots, chat history, and text timelines |
| **UI** | Streamlit-only — no FastAPI, no Flask — with cached detection, per-provider keys, live camera support, and persistent report/export state |

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) [Ollama](https://ollama.com/) for local LLM inference
- (Optional) Webcam for live camera mode

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd "Agentic Vision Intelligence Platform"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in API keys as needed:

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

| Variable | Required | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | No | Ollama server URL (default: `http://localhost:11434`) |
| `OPENAI_API_KEY` | If using OpenAI | OpenAI API key |
| `GEMINI_API_KEY` | If using Gemini | Google Gemini API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | Anthropic API key |

If you prefer not to store cloud keys in `.env`, you can enter them directly in the Streamlit sidebar. They remain in the current app session only.

### Run the Application

```bash
streamlit run src/ui/streamlit_app.py
```

Recommended local-first workflow:

1. Start with `Ollama` as the provider.
2. Fetch local models or enter one manually.
3. Initialise the pipeline for CV-only or CV+LLM mode.
4. Switch to cloud providers only when you want stronger reasoning/reporting quality.

### Run Tests

```bash
pytest tests/ -v
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed module documentation.

### Project Structure

```
├── config/                  # Pydantic Settings, constants
│   └── __init__.py
├── src/
│   ├── input/               # Input source abstraction (camera, image, video)
│   ├── vision/              # YOLO26 detector, event extractor
│   ├── memory/              # Scene state, event timeline, session store
│   ├── providers/           # LLM provider adapters (Ollama, OpenAI, Gemini, Anthropic)
│   ├── reasoning/           # Prompt templates, Reasoner wrapper
│   ├── orchestration/       # LangGraph workflows and node functions
│   ├── reporting/           # Alert management, session export
│   ├── utils/               # Frame processing utilities
│   └── ui/                  # Streamlit application
├── tests/                   # Unit tests (pytest)
├── data/                    # Sample input data
├── output/                  # Session exports
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
└── README.md
```

### YOLO26 Variants

| Variant | Weight File | Default Mode | Use Case |
|---|---|---|---|
| YOLO26n | `yolo26n.pt` | Live camera | Low latency, real-time |
| YOLO26m | `yolo26m.pt` | Video | Balanced speed/quality |
| YOLO26l | `yolo26l.pt` | Image | Highest accuracy |

### LLM Providers

All providers implement the same `LLMProvider` interface:
- `list_models()` — dynamically fetch available models
- `generate()` — text + optional image inference
- `is_available()` — connectivity check

If model listing fails, the app degrades gracefully to manual model entry instead of blocking the workflow.

## Exported Session Artefacts

Each export writes a session folder under `output/` containing:

- `events.json`
- `alerts.json`
- `summary.json`
- `report.md` when a report has been generated
- `config_snapshot.json`
- `chat_history.json`
- `event_timeline.txt`

## CI/CD

```bash
# Lint
ruff check .

# Type check
mypy src/ --ignore-missing-imports

# Tests with coverage
pytest tests/ -v --cov=src --cov=config --cov-report=term-missing
```

## License

MIT
