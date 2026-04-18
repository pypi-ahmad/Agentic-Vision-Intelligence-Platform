"""Agentic Vision Intelligence Platform — Streamlit UI (single entry-point)."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to path so imports work when Streamlit runs the file directly.
# Must happen before ``config`` / ``src.*`` imports below — linted exceptions
# are scoped in ``pyproject.toml`` (``[tool.ruff.lint.per-file-ignores]``).
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    MODE_DEFAULT_MODELS,
    MODE_MAX_DIM,
    MODE_USES_TRACKING,
    PROVIDERS,
    VIDEO_DISPLAY_INTERVAL,
    YOLO_MODELS,
    get_settings,
)
from src.input.camera import CameraSource
from src.input.image import ImageArraySource
from src.input.video import VideoSource
from src.memory.session_store import SessionStore
from src.orchestration import (
    ask_question,
    clear_reasoner,
    generate_report,
    get_event_timeline,
    get_reasoner,
    get_scene_state,
    process_frame,
    reset_session,
    set_detector,
    set_reasoner,
)
from src.providers import get_provider
from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from src.reporting.alerts import AlertManager
from src.reporting.exporter import SessionExporter
from src.utils.frame_utils import bgr_to_rgb, pil_to_numpy, resize_frame, rgb_to_bgr

# --- Logging configuration --------------------------------------------
# Users can opt into verbose logging via ``LOG_LEVEL=DEBUG`` in ``.env``
# or the shell environment; default is INFO so Streamlit's own logs stay
# readable.  Called once at import time — Streamlit re-imports the script
# on each rerun, but ``basicConfig`` is a no-op if a root handler exists.
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ======================================================================
# PAGE CONFIG
# ======================================================================

st.set_page_config(
    page_title="Agentic Vision Intelligence Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================================
# SESSION STATE DEFAULTS
# ======================================================================

def _init_state():
    started_at = datetime.now()
    defaults = {
        "mode": "image",
        "yolo_variant": MODE_DEFAULT_MODELS.get("image", "YOLO26l"),
        "confidence": 0.35,
        "provider_name": "Ollama",
        "api_keys": {"OpenAI": "", "Gemini": "", "Anthropic": ""},
        "selected_models": {provider: "" for provider in PROVIDERS},
        "ollama_url": "http://localhost:11434",
        "available_models": [],
        "models_provider": "",
        "detector_ready": False,
        "reasoner_ready": False,
        "alert_manager": AlertManager(),
        "chat_history": [],
        "last_result": None,
        "session_id": started_at.strftime("%Y%m%d_%H%M%S"),
        "session_started_at": started_at,
        "frame_index": 0,
        "last_export_dir": "",
        "_last_provider_name": "Ollama",
        "_last_report": "",
        "_img_cache_key": "",
        "_model_fetch_error": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ======================================================================
# SIDEBAR — Configuration Controls
# ======================================================================

def _get_active_api_key() -> str:
    pn = st.session_state.provider_name
    if pn == "Ollama":
        return ""
    return st.session_state.api_keys.get(pn, "")


def _get_active_model() -> str:
    return st.session_state.selected_models.get(st.session_state.provider_name, "")


def _set_active_model(model_name: str) -> None:
    st.session_state.selected_models[st.session_state.provider_name] = model_name


def _session_duration_text() -> str:
    started_at = st.session_state.session_started_at
    elapsed = max(0, int((datetime.now() - started_at).total_seconds()))
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption(f"Session `{st.session_state.session_id}`  •  Running {_session_duration_text()}")

        # ---- Input Mode ------------------------------------------------
        st.subheader("Input Source")
        mode = st.radio(
            "Mode",
            options=["image", "video", "live"],
            index=["image", "video", "live"].index(st.session_state.mode),
            horizontal=True,
            key="mode_radio",
            help="**Image**: single-frame analysis  •  **Video**: batch processing with tracking  •  **Live**: real-time camera feed",
        )
        if mode != st.session_state.mode:
            st.session_state.mode = mode
            # Auto-select default YOLO variant for new mode
            st.session_state.yolo_variant = MODE_DEFAULT_MODELS.get(mode, "YOLO26n")
            st.session_state.detector_ready = False

        # ---- CV Model --------------------------------------------------
        st.subheader("CV Model")
        variant_names = list(YOLO_MODELS.keys())
        mode_default = MODE_DEFAULT_MODELS.get(st.session_state.mode, "YOLO26n")
        default_idx = variant_names.index(st.session_state.yolo_variant) if st.session_state.yolo_variant in variant_names else 0
        chosen_variant = st.selectbox(
            "YOLO26 Variant",
            options=variant_names,
            index=default_idx,
            help=f"n=fast/nano, m=balanced, l=accurate/large  •  Recommended for **{st.session_state.mode}**: **{mode_default}**",
        )
        if chosen_variant != st.session_state.yolo_variant:
            st.session_state.yolo_variant = chosen_variant
            st.session_state.detector_ready = False

        confidence = st.slider("Confidence", 0.05, 1.0, st.session_state.confidence, 0.05)
        st.session_state.confidence = confidence

        # ---- LLM Provider ----------------------------------------------
        st.subheader("LLM Provider")
        provider_name = st.selectbox("Provider", PROVIDERS, index=PROVIDERS.index(st.session_state.provider_name))
        if provider_name != st.session_state._last_provider_name:
            st.session_state.provider_name = provider_name
            st.session_state.available_models = []
            st.session_state.models_provider = ""
            st.session_state._last_provider_name = provider_name
            st.session_state.reasoner_ready = False
            st.session_state._model_fetch_error = ""
        else:
            st.session_state.provider_name = provider_name

        if provider_name == "Ollama":
            st.session_state.ollama_url = st.text_input("Ollama URL", st.session_state.ollama_url)
            st.caption("Local-first mode. No API key is required.")
        else:
            stored_key = st.session_state.api_keys.get(provider_name, "")
            key = st.text_input(
                f"{provider_name} API Key",
                value=stored_key,
                type="password",
                key=f"key_{provider_name}",
            )
            st.session_state.api_keys[provider_name] = key
            st.caption("API keys stay in this Streamlit session only unless you put them in .env yourself.")

        # Fetch models button
        if st.button("🔄 Fetch Models"):
            _fetch_models()

        # --- Show fetch error feedback --------------------------------
        _err = st.session_state.get("_model_fetch_error", "")
        if _err == "auth":
            st.error(f"🔑 Missing or invalid API key for {provider_name}.")
        elif _err == "connection":
            st.error(f"🌐 Cannot reach {provider_name} — check your network or URL.")
        elif _err == "rate_limit":
            st.warning(f"⏳ {provider_name} rate-limited your request. Wait a moment and retry.")
        elif _err == "empty":
            st.warning(f"No models returned by {provider_name}. The service may be temporarily unavailable.")
        elif _err == "error":
            st.error("Provider error — see logs for details.")

        # --- Model selector or manual fallback ------------------------
        if st.session_state.available_models and st.session_state.models_provider == provider_name:
            idx = 0
            active_model = _get_active_model()
            if active_model in st.session_state.available_models:
                idx = st.session_state.available_models.index(active_model)
            selected_model = st.selectbox(
                "Model", st.session_state.available_models, index=idx
            )
            _set_active_model(selected_model)
        else:
            # Fallback: manual model name entry when fetch unavailable
            manual = st.text_input(
                "Model name",
                value=_get_active_model(),
                help="Enter a model name manually, or click **Fetch Models** to load the list.",
            )
            _set_active_model(manual)

        # ---- Initialise Button ------------------------------------------
        st.divider()
        if st.button("🚀 Initialise Pipeline", type="primary", width="stretch"):
            _initialise_pipeline()

        st.markdown("##### Pipeline Status")
        if not st.session_state.detector_ready:
            st.info("Select your settings above and click **Initialise Pipeline** to begin analysis.")
        elif st.session_state.reasoner_ready:
            st.success(f"✔ CV: {st.session_state.yolo_variant}  •  LLM: {st.session_state.provider_name} / {_get_active_model()}")
        else:
            st.warning(f"✔ CV: {st.session_state.yolo_variant}  •  ✗ LLM not configured — Q&A and reports unavailable")

        # ---- Session Controls -------------------------------------------
        st.divider()
        if st.button("🔄 Reset Session", width="stretch"):
            reset_session()
            st.session_state.alert_manager.reset()
            st.session_state.chat_history.clear()
            st.session_state.last_result = None
            st.session_state.frame_index = 0
            st.session_state.detector_ready = False
            st.session_state.reasoner_ready = False
            st.session_state.last_export_dir = ""
            st.session_state._last_report = ""
            st.session_state._img_cache_key = ""
            st.session_state._model_fetch_error = ""
            st.rerun()


def _fetch_models():
    """Fetch model list from the active provider with typed error handling."""
    st.session_state._model_fetch_error = ""
    prov_name = st.session_state.provider_name

    # Pre-check: cloud providers need a key before we even try
    if prov_name != "Ollama" and not _get_active_api_key().strip():
        st.session_state._model_fetch_error = "auth"
        st.session_state.available_models = []
        st.session_state.models_provider = ""
        return

    try:
        prov = _make_provider()
    except ProviderAuthError:
        st.session_state._model_fetch_error = "auth"
        st.session_state.available_models = []
        st.session_state.models_provider = ""
        return

    try:
        with st.spinner(f"Fetching models from {prov_name}…"):
            models = prov.list_models()
    except ProviderAuthError:
        st.session_state._model_fetch_error = "auth"
        st.session_state.available_models = []
        st.session_state.models_provider = ""
        return
    except ProviderConnectionError:
        st.session_state._model_fetch_error = "connection"
        st.session_state.available_models = []
        st.session_state.models_provider = ""
        return
    except ProviderRateLimitError:
        st.session_state._model_fetch_error = "rate_limit"
        st.session_state.available_models = []
        st.session_state.models_provider = ""
        return
    except ProviderError as exc:
        # S1: never forward the raw SDK exception message into session state
        # (may include URLs / headers / partial bodies).  Log full detail for
        # the operator, show a stable, curated marker to the UI.
        logger.error("Provider error fetching models from %s", prov_name, exc_info=exc)
        st.session_state._model_fetch_error = "error"
        st.session_state.available_models = []
        st.session_state.models_provider = ""
        return

    st.session_state.available_models = models
    st.session_state.models_provider = prov_name
    if models:
        # Preserve previous selection if it's still in the list
        prev = _get_active_model()
        if prev not in models:
            _set_active_model(models[0])
        st.toast(f"Found {len(models)} models from {prov_name}", icon="✅")
    else:
        st.session_state._model_fetch_error = "empty"


def _make_provider() -> LLMProvider:
    return get_provider(
        st.session_state.provider_name,
        api_key=_get_active_api_key(),
        ollama_url=st.session_state.ollama_url,
    )


def _initialise_pipeline():
    try:
        set_detector(st.session_state.yolo_variant, st.session_state.confidence)
        st.session_state.detector_ready = True
        st.session_state.reasoner_ready = False
        model_name = _get_active_model().strip()
        if model_name:
            prov = _make_provider()
            set_reasoner(prov, model_name)
            st.session_state.reasoner_ready = True
        else:
            clear_reasoner()
        st.toast("Pipeline initialised!", icon="🚀")
    except ProviderAuthError as exc:
        st.session_state.reasoner_ready = False
        logger.error("LLM auth failed during initialise", exc_info=exc)
        st.error("🔑 LLM auth failed — check your API key.")
    except ProviderConnectionError as exc:
        st.session_state.reasoner_ready = False
        logger.error("Cannot reach LLM provider during initialise", exc_info=exc)
        st.error("🌐 Cannot reach LLM provider — check your network or URL.")
    except Exception as exc:
        st.session_state.reasoner_ready = False
        logger.error("Pipeline initialise failed", exc_info=exc)
        st.error("Init failed — see logs for details.")


# ======================================================================
# MAIN WORKSPACE
# ======================================================================

def _render_main():
    st.title("🔍 Agentic Vision Intelligence Platform")
    mode = st.session_state.mode
    variant = st.session_state.yolo_variant
    recommended = MODE_DEFAULT_MODELS.get(mode, "YOLO26n")
    override_note = "" if variant == recommended else f"  _(default: {recommended})_"
    tracking = "Tracking" if MODE_USES_TRACKING.get(mode, False) else "Detection"
    max_res = MODE_MAX_DIM.get(mode, 1280)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Mode** &nbsp; `{mode}`")
    c2.markdown(f"**Model** &nbsp; `{variant}`{override_note}")
    c3.markdown(f"**Pipeline** &nbsp; {tracking} @ {max_res}px")
    c4.markdown(f"**Session** &nbsp; {_session_duration_text()}")

    if not st.session_state.detector_ready:
        st.info("Open the sidebar to configure your detection model and LLM provider, then click **Initialise Pipeline** to begin.")
        return

    if mode == "image":
        _render_image_mode()
    elif mode == "video":
        _render_video_mode()
    elif mode == "live":
        _render_live_mode()


# ---- Image Mode ------------------------------------------------------

def _render_image_mode():
    """Image mode — YOLO26l default, max quality, single-frame detect()."""
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded is None:
        return

    max_dim = MODE_MAX_DIM.get("image", 1280)
    pil_img = Image.open(uploaded).convert("RGB")
    display_frame = resize_frame(pil_to_numpy(pil_img), max_dim=max_dim)
    source = ImageArraySource(rgb_to_bgr(display_frame), name=uploaded.name)
    source.open()
    packet = source.read()
    source.close()
    if packet is None:
        st.error("Failed to create image packet.")
        return

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown("#### Original")
        st.image(display_frame, width="stretch")

    # Only re-run CV when the image actually changes (avoid re-detection on every rerun)
    cache_key = f"{uploaded.name}_{uploaded.size}_{st.session_state.yolo_variant}_{st.session_state.confidence}"
    if st.session_state.get("_img_cache_key") != cache_key:
        with st.spinner("Running detection…"):
            result = process_frame(packet.frame, mode="image", source_id=packet.source_id, frame_index=packet.frame_index)
        st.session_state.last_result = result
        st.session_state["_img_cache_key"] = cache_key
        _ingest_alerts(result)
    else:
        result = st.session_state.last_result

    with col_out:
        st.markdown("#### Detection Results")
        ann = result.get("annotated_frame")
        if ann is not None:
            st.image(bgr_to_rgb(ann), width="stretch")

    _render_result_tabs(result)


# ---- Video Mode -------------------------------------------------------

def _render_video_mode():
    """Video mode — YOLO26m default, balanced, track() with display throttling."""
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        return

    # C9: write uploads into the session's output directory (same mount as
    # exports) instead of %TEMP% so a crash / browser refresh leaves the file
    # in a predictable, user-discoverable location and clean-up with
    # ``Path.unlink(missing_ok=True)`` is sufficient on Windows.
    cfg = get_settings()
    max_dim = MODE_MAX_DIM.get("video", 960)
    tmp_dir = cfg.output_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"upload_{uuid.uuid4().hex}.mp4"
    tmp_path.write_bytes(uploaded.getbuffer())
    try:
        source = VideoSource(str(tmp_path))
        source.open()
        total_frames = source.total_frames
        fps = source.fps

        sample_rate = cfg.frame_sample_rate
        display_interval = VIDEO_DISPLAY_INTERVAL

        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Total Frames", f"{total_frames:,}")
        vc2.metric("Source FPS", f"{fps:.1f}")
        vc3.metric("Sample Rate", f"1 / {sample_rate}")
        st.caption(f"UI refreshes every {display_interval} processed frames for performance.")

        progress = st.progress(0.0)
        frame_display = st.empty()
        status_display = st.empty()
        last_result = None
        processed = 0

        while True:
            packet = source.read()
            if packet is None:
                break
            if packet.frame_index % sample_rate == 0:
                frame_bgr = resize_frame(packet.frame, max_dim=max_dim)
                # P1/QW8: only annotate the frames we're about to display.
                display_tick = (processed + 1) % display_interval == 0 or (
                    packet.frame_index + sample_rate >= total_frames
                )
                result = process_frame(
                    frame_bgr,
                    mode="video",
                    source_id=packet.source_id,
                    frame_index=packet.frame_index,
                    draw=display_tick,
                )
                last_result = result
                _ingest_alerts(result)
                processed += 1
                if display_tick:
                    ann = result.get("annotated_frame")
                    if ann is not None:
                        frame_display.image(bgr_to_rgb(ann), width="stretch")
                    status_display.caption(f"Frame {packet.frame_index}/{total_frames} — {result.get('detection_summary', '')}")
                    progress.progress(min(packet.frame_index / max(total_frames, 1), 1.0))

        source.close()
        progress.progress(1.0)
        status_display.success(f"Processing complete — {processed} frames analysed out of {total_frames} total.")

        if last_result:
            st.session_state.last_result = last_result
            _render_result_tabs(last_result)
    finally:
        # Best-effort clean-up; if the file was already removed or never
        # created, ``missing_ok=True`` silently succeeds.
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("Unable to remove temporary video file %s: %s", tmp_path, exc)



# ---- Live Camera Mode ------------------------------------------------

def _get_or_open_camera() -> CameraSource | None:
    """Return a camera source kept alive across reruns.

    Opening ``cv2.VideoCapture`` on Windows' DShow/MSMF backend can cost
    500\u20133000 ms, so we stash a single instance on ``st.session_state`` and
    only open it once per "Camera Feed" toggle cycle.  Failure is surfaced
    via ``st.error`` and leaves the slot empty so the fragment simply stops.
    """
    cam = st.session_state.get("_live_cam_source")
    if cam is not None:
        return cam
    try:
        cfg = get_settings()
        cam = CameraSource(camera_index=cfg.camera_index)
        cam.open()
    except Exception as exc:  # pragma: no cover \u2014 depends on hardware
        logger.error("Cannot open camera", exc_info=exc)
        st.error("Cannot open camera. See logs for details.")
        st.session_state._live_cam_source = None
        return None
    st.session_state._live_cam_source = cam
    return cam


def _release_camera() -> None:
    cam = st.session_state.get("_live_cam_source")
    if cam is not None:
        try:
            cam.close()
        except Exception:  # pragma: no cover
            pass
        st.session_state._live_cam_source = None


def _render_live_mode():
    """Live mode \u2014 YOLO26n default, low-latency track(), suppressed auto-LLM.

    C3/P3/N1: the camera is opened once and the per-frame render loop is
    hosted in an ``st.fragment`` that re-runs on its own timer instead of
    via full-script ``st.rerun()``, eliminating the open/close-per-rerun
    pattern that was unusable on Windows.
    """
    col_ctrl, col_info = st.columns([1, 3])
    with col_ctrl:
        run = st.toggle("\u25b6 Camera Feed", value=False, key="_live_cam_toggle")
    with col_info:
        if run:
            st.caption(
                "Camera active \u2014 low-latency pipeline. "
                "Periodic summaries suppressed; use the Q&A tab for on-demand analysis."
            )

    if not run:
        _release_camera()
        st.info("Enable the camera toggle above to start real-time analysis.")
        if st.session_state.last_result:
            _render_result_tabs(st.session_state.last_result)
        return

    cam = _get_or_open_camera()
    if cam is None:
        return

    _live_camera_fragment()

    if st.session_state.last_result:
        _render_result_tabs(st.session_state.last_result)


@st.fragment(run_every=0.1)
def _live_camera_fragment() -> None:
    """Inner render loop \u2014 fragment re-runs only this block, not the whole app."""
    cam: CameraSource | None = st.session_state.get("_live_cam_source")
    if cam is None:
        return

    cfg = get_settings()
    max_dim = MODE_MAX_DIM.get("live", 640)

    # Drain up to ``skip_count`` buffered frames so we show the freshest image.
    packet = None
    skip_count = max(1, min(cfg.frame_sample_rate, 3))
    for _ in range(skip_count):
        nxt = cam.read()
        if nxt is None:
            break
        packet = nxt

    if packet is None:
        st.warning("Camera read failed.")
        return

    frame_bgr = resize_frame(packet.frame, max_dim=max_dim)
    # M6: the session counter advances monotonically; packet.frame_index is
    # per-``open()`` and now unused in the emitted frame number.
    st.session_state.frame_index += 1
    fi = st.session_state.frame_index
    result = process_frame(
        frame_bgr,
        mode="live",
        source_id=packet.source_id,
        frame_index=fi,
        draw=True,
    )
    st.session_state.last_result = result
    _ingest_alerts(result)

    ann = result.get("annotated_frame")
    if ann is not None:
        st.image(bgr_to_rgb(ann), width="stretch")
    st.caption(f"Frame {fi} \u2014 {result.get('detection_summary', '')}")


# ======================================================================
# RESULT DISPLAY
# ======================================================================

def _ingest_alerts(result: dict):
    new_evts = result.get("new_events", [])
    st.session_state.alert_manager.ingest_events(new_evts)


def _render_result_tabs(result: dict):
    tabs = st.tabs(["📊 Summary", "🔔 Events", "⚠️ Alerts", "💬 Q&A", "📝 Report", "💾 Export"])

    # ---- Summary
    with tabs[0]:
        scene = get_scene_state()
        timeline = get_event_timeline()
        scene_summary = scene.get_summary()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Objects Detected", sum(result.get("object_counts", {}).values()))
        col2.metric("Active Tracks", scene_summary.get("active_tracked", 0))
        col3.metric("Events Recorded", timeline.count)
        col4.metric("Pending Alerts", len(st.session_state.alert_manager.unacknowledged))
        st.markdown(f"**Detection Summary:** {result.get('detection_summary', 'No detections')}")
        st.caption(scene.get_description())
        counts = result.get("object_counts", {})
        if counts:
            st.bar_chart(counts)
        if result.get("llm_response") and result.get("reasoning_task") in {"summarize", "anomaly", "describe"}:
            st.markdown("---")
            st.markdown("**🤖 Agent Insight**")
            st.info(result.get("llm_response"))
        with st.expander("Scene State Details", expanded=False):
            st.json(scene_summary)

    # ---- Events
    with tabs[1]:
        tl = get_event_timeline()
        if tl.count:
            st.markdown(f"**{tl.count} event(s)** recorded this session.")
            st.text(tl.to_text())
            with st.expander("Event Statistics", expanded=False):
                st.json(tl.get_summary())
        else:
            st.info("No events recorded yet. Events appear as objects enter, exit, or change state in the scene.")

    # ---- Alerts
    with tabs[2]:
        am: AlertManager = st.session_state.alert_manager
        unack = am.unacknowledged
        if unack:
            for a in unack:
                with st.expander(f"[{a.severity.upper()}] {a.description}", expanded=True):
                    st.json(a.to_dict())
                    if st.button("Acknowledge", key=f"ack_{a.alert_id}"):
                        am.acknowledge(a.alert_id)
                        st.rerun()
        else:
            st.info("No pending alerts. Alerts are generated when significant scene changes or anomalies are detected.")

    # ---- Q&A
    with tabs[3]:
        _render_qa()

    # ---- Report
    with tabs[4]:
        _render_report()

    # ---- Export
    with tabs[5]:
        _render_export()


def _render_qa():
    if get_reasoner() is None:
        st.warning("An LLM provider is required for Q&A. Configure one in the sidebar and click **Initialise Pipeline**.")
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about the current scene…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                last = st.session_state.last_result or {}
                frame = last.get("current_frame")
                res = ask_question(question, frame=frame, mode=st.session_state.mode)
                answer = res.get("answer", res.get("llm_response", ""))
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


def _render_report():
    if get_reasoner() is None:
        st.warning("An LLM provider is required for report generation. Configure one in the sidebar and click **Initialise Pipeline**.")
        return

    if st.button("📝 Generate Report", width="stretch"):
        with st.spinner("Generating report…"):
            res = generate_report(session_id=st.session_state.session_id, duration=_session_duration_text())
            report = res.get("report", res.get("llm_response", ""))
        st.session_state["_last_report"] = report

    if st.session_state.get("_last_report"):
        st.markdown(st.session_state["_last_report"])
    else:
        st.info("Click **Generate Report** to create a comprehensive summary of session activity, detections, and events.")


def _render_export():
    if st.button("💾 Export Session", width="stretch"):
        exp = SessionExporter(session_id=st.session_state.session_id)
        store = SessionStore(session_id=st.session_state.session_id)

        tl = get_event_timeline()
        exp.save_events(tl.to_list())
        exp.save_alerts(st.session_state.alert_manager.to_list())
        exp.save_summary(get_scene_state().get_summary())
        if st.session_state.get("_last_report"):
            exp.save_report(st.session_state["_last_report"])
        store.save_json(
            "config_snapshot.json",
            {
                "mode": st.session_state.mode,
                "yolo_variant": st.session_state.yolo_variant,
                "confidence": st.session_state.confidence,
                "provider": st.session_state.provider_name,
                "model": _get_active_model(),
                "session_id": st.session_state.session_id,
                "duration": _session_duration_text(),
            },
        )
        store.save_json("chat_history.json", st.session_state.chat_history)
        store.save_text("event_timeline.txt", tl.to_text())
        st.session_state.last_export_dir = str(exp.export_dir)

        st.success(f"Session data exported to `{exp.export_dir}`")

    if st.session_state.last_export_dir:
        st.caption(f"Last export location: `{st.session_state.last_export_dir}`")


# ======================================================================
# ENTRY POINT
# ======================================================================

_render_sidebar()
_render_main()
