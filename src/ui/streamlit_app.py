"""Agentic Vision Intelligence Platform — Streamlit UI (single entry-point)."""

from __future__ import annotations

from datetime import datetime
import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to path so imports work when Streamlit runs the file directly
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    PROVIDERS,
    YOLO_MODELS,
    MODE_DEFAULT_MODELS,
    MODE_MAX_DIM,
    MODE_USES_TRACKING,
    VIDEO_DISPLAY_INTERVAL,
    get_settings,
)
from src.input.camera import CameraSource
from src.input.image import ImageArraySource
from src.input.video import VideoSource
from src.memory.session_store import SessionStore
from src.providers import get_provider
from src.providers.base import (
    LLMProvider,
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
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
from src.reporting.alerts import AlertManager
from src.reporting.exporter import SessionExporter
from src.utils.frame_utils import bgr_to_rgb, pil_to_numpy, resize_frame, rgb_to_bgr

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
        st.caption(f"Session: {st.session_state.session_id}")

        # ---- Input Mode ------------------------------------------------
        st.subheader("Input Mode")
        mode = st.radio(
            "Source",
            options=["image", "video", "live"],
            index=["image", "video", "live"].index(st.session_state.mode),
            horizontal=True,
            key="mode_radio",
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
        elif _err.startswith("error:"):
            st.error(f"Provider error: {_err[6:]}")

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
        if st.button("🚀 Initialise Pipeline", type="primary", use_container_width=True):
            _initialise_pipeline()

        if not st.session_state.detector_ready:
            st.info("Configure and click **Initialise Pipeline**.")
        elif st.session_state.reasoner_ready:
            st.success(f"CV ready: {st.session_state.yolo_variant} • LLM ready: {st.session_state.provider_name} / {_get_active_model()}")
        else:
            st.warning(f"CV ready: {st.session_state.yolo_variant} • LLM not configured")

        # ---- Session Controls -------------------------------------------
        st.divider()
        if st.button("🔄 Reset Session", use_container_width=True):
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
        st.session_state._model_fetch_error = f"error:{exc}"
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
        st.error(f"🔑 LLM auth failed: {exc}")
    except ProviderConnectionError as exc:
        st.session_state.reasoner_ready = False
        st.error(f"🌐 Cannot reach LLM provider: {exc}")
    except Exception as exc:
        st.session_state.reasoner_ready = False
        st.error(f"Init failed: {exc}")


# ======================================================================
# MAIN WORKSPACE
# ======================================================================

def _render_main():
    st.title("🔍 Agentic Vision Intelligence Platform")
    mode = st.session_state.mode
    variant = st.session_state.yolo_variant
    recommended = MODE_DEFAULT_MODELS.get(mode, "YOLO26n")
    override_note = "" if variant == recommended else f" (override — default: {recommended})"
    tracking = "tracking" if MODE_USES_TRACKING.get(mode, False) else "detect"
    st.caption(
        f"Mode: **{mode}** • CV: {variant}{override_note} [{tracking}] "
        f"• Resize: {MODE_MAX_DIM.get(mode, 1280)}px • Runtime: {_session_duration_text()}"
    )

    if not st.session_state.detector_ready:
        st.info("👈 Configure the pipeline in the sidebar and click **Initialise Pipeline** to begin.")
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
        st.subheader("Input")
        st.image(display_frame, use_container_width=True)

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
        st.subheader("Detections")
        ann = result.get("annotated_frame")
        if ann is not None:
            st.image(bgr_to_rgb(ann), use_container_width=True)

    _render_result_tabs(result)


# ---- Video Mode -------------------------------------------------------

def _render_video_mode():
    """Video mode — YOLO26m default, balanced, track() with display throttling."""
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        return

    max_dim = MODE_MAX_DIM.get("video", 960)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tfile.name
    try:
        tfile.write(uploaded.read())
        tfile.flush()
        tfile.close()
        source = VideoSource(tmp_path)
        source.open()
        total_frames = source.total_frames
        fps = source.fps

        cfg = get_settings()
        sample_rate = cfg.frame_sample_rate
        display_interval = VIDEO_DISPLAY_INTERVAL

        st.info(
            f"Video: {total_frames} frames @ {fps:.1f} FPS — "
            f"sampling every {sample_rate} frame(s), "
            f"UI refresh every {display_interval} processed frames"
        )

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
                result = process_frame(frame_bgr, mode="video", source_id=packet.source_id, frame_index=packet.frame_index)
                last_result = result
                _ingest_alerts(result)
                processed += 1
                # Throttle UI updates — only render every N processed frames
                if processed % display_interval == 0 or packet.frame_index + sample_rate >= total_frames:
                    ann = result.get("annotated_frame")
                    if ann is not None:
                        frame_display.image(bgr_to_rgb(ann), use_container_width=True)
                    status_display.caption(f"Frame {packet.frame_index}/{total_frames} — {result.get('detection_summary', '')}")
                    progress.progress(min(packet.frame_index / max(total_frames, 1), 1.0))

        source.close()
        progress.progress(1.0)
        status_display.success(f"Processed {processed} frames from {total_frames} total.")

        if last_result:
            st.session_state.last_result = last_result
            _render_result_tabs(last_result)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError as exc:
            logger.debug("Unable to remove temporary video file %s: %s", tmp_path, exc)



# ---- Live Camera Mode ------------------------------------------------

def _render_live_mode():
    """Live mode — YOLO26n default, low-latency track(), suppressed auto-LLM."""
    col_ctrl, col_info = st.columns([1, 3])
    with col_ctrl:
        run = st.toggle("▶ Start Camera", value=False)
    with col_info:
        if run:
            st.caption(
                "Camera active — YOLO26n optimised path. "
                "Auto-LLM summaries suppressed; use Q&A for on-demand analysis."
            )

    if not run:
        st.info("Toggle the switch above to start the live camera feed.")
        # Show last result tabs if we have data from previous frames
        if st.session_state.last_result:
            _render_result_tabs(st.session_state.last_result)
        return

    cfg = get_settings()
    max_dim = MODE_MAX_DIM.get("live", 640)
    frame_holder = st.empty()
    info_holder = st.empty()

    source = CameraSource(camera_index=cfg.camera_index)
    try:
        source.open()
    except Exception as exc:
        st.error(f"Cannot open camera: {exc}")
        return

    # Read the latest frame, skip stale buffered frames for freshness
    packet = None
    skip_count = min(cfg.frame_sample_rate, 3)  # cap skips to keep cycle fast
    for _ in range(max(1, skip_count)):
        packet = source.read()
        if packet is None:
            break
    source.close()

    if packet is None:
        st.warning("Camera read failed.")
        return

    # Resize to smaller dim for faster YOLO inference in live mode
    frame_bgr = resize_frame(packet.frame, max_dim=max_dim)
    result = process_frame(frame_bgr, mode="live", source_id=packet.source_id, frame_index=packet.frame_index + st.session_state.frame_index)
    st.session_state.last_result = result
    st.session_state.frame_index += max(1, skip_count)
    _ingest_alerts(result)
    ann = result.get("annotated_frame")
    if ann is not None:
        frame_holder.image(bgr_to_rgb(ann), use_container_width=True)
    info_holder.caption(f"Frame {st.session_state.frame_index} — {result.get('detection_summary', '')}")

    _render_result_tabs(result)

    # Trigger next cycle
    st.rerun()


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
        col1.metric("Detections", sum(result.get("object_counts", {}).values()))
        col2.metric("Tracked", scene_summary.get("active_tracked", 0))
        col3.metric("Events", timeline.count)
        col4.metric("Alerts", len(st.session_state.alert_manager.unacknowledged))
        st.markdown(f"**Detection:** {result.get('detection_summary', 'N/A')}")
        st.caption(scene.get_description())
        counts = result.get("object_counts", {})
        if counts:
            st.bar_chart(counts)
        if result.get("llm_response") and result.get("reasoning_task") in {"summarize", "anomaly", "describe"}:
            st.markdown("**Agent Insight**")
            st.write(result.get("llm_response"))
        st.json(scene_summary)

    # ---- Events
    with tabs[1]:
        tl = get_event_timeline()
        if tl.count:
            st.text(tl.to_text())
            st.json(tl.get_summary())
        else:
            st.info("No events recorded yet.")

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
            st.info("No unacknowledged alerts.")

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
        st.warning("Select and initialise an LLM provider to use Q&A.")
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about the scene…")
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
        st.warning("Select and initialise an LLM provider to generate reports.")
        return

    if st.button("📝 Generate Report", use_container_width=True):
        with st.spinner("Generating report…"):
            res = generate_report(session_id=st.session_state.session_id, duration=_session_duration_text())
            report = res.get("report", res.get("llm_response", ""))
        st.session_state["_last_report"] = report

    if st.session_state.get("_last_report"):
        st.markdown(st.session_state["_last_report"])
    else:
        st.info("Generate a report once you have enough scene history and events.")


def _render_export():
    if st.button("💾 Export Session", use_container_width=True):
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

        st.success(f"Session exported to {exp.export_dir}")

    if st.session_state.last_export_dir:
        st.caption(f"Last export: {st.session_state.last_export_dir}")


# ======================================================================
# ENTRY POINT
# ======================================================================

_render_sidebar()
_render_main()
