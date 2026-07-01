import logging
import sys
import time
from pathlib import Path

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import VoicePipeline, TurnResult
from src.wake_word import WakeWordDetector

st.set_page_config(
    page_title="Nova AI Voice Agent",
    page_icon="🎙️",
    layout="centered",
)

# -------------------------------------------------------------------
# Dark Mode state
# -------------------------------------------------------------------

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
dark_mode = st.session_state.dark_mode

theme_bg = "#0f0c29" if dark_mode else "#f5f0ff"
theme_bg2 = "#302b63" if dark_mode else "#e8dff5"
theme_card_user = "rgba(161,140,209,0.12)" if dark_mode else "rgba(161,140,209,0.08)"
theme_card_ai = "rgba(251,194,235,0.10)" if dark_mode else "rgba(251,194,235,0.06)"
theme_text = "rgba(255,255,255,0.88)" if dark_mode else "#1a1a2e"
theme_text_muted = "rgba(255,255,255,0.5)" if dark_mode else "rgba(0,0,0,0.45)"
theme_border = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.08)"
theme_status_bg = "rgba(255,255,255,0.07)" if dark_mode else "rgba(0,0,0,0.04)"
theme_btn_sec = "rgba(255,255,255,0.08)" if dark_mode else "rgba(0,0,0,0.06)"
theme_btn_sec_hover = "rgba(255,255,255,0.14)" if dark_mode else "rgba(0,0,0,0.10)"

# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------

st.markdown(f"""
<style>
    .main > div {{ padding-bottom: 2rem; }}
    .stApp {{ background: linear-gradient(135deg, {theme_bg}, {theme_bg2}, {theme_bg}); }}

    .nova-header {{
        text-align: center;
        padding: 1.5rem 0 0.25rem 0;
    }}
    .nova-header h1 {{
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a18cd1, #fbc2eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }}
    .nova-header p {{
        color: {theme_text_muted};
        font-size: 0.85rem;
        margin-top: 0.15rem;
    }}

    /* ── Mic button ────────────────────────────────── */
    .mic-container {{
        display: flex;
        justify-content: center;
        margin: 1.2rem 0 0.4rem 0;
    }}
    .mic-container .stButton button {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        font-size: 2rem;
        border: none;
        transition: all 0.25s ease;
        background: linear-gradient(135deg, #a18cd1, #fbc2eb);
        box-shadow: 0 4px 24px rgba(161,140,209,0.35);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        line-height: 1;
    }}
    .mic-container .stButton button[kind="primary"]:hover {{
        transform: scale(1.06);
        box-shadow: 0 6px 32px rgba(161,140,209,0.5);
    }}
    .mic-container .stButton button[kind="primary"]:active {{
        transform: scale(0.96);
    }}
    .mic-container .stButton button:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
    }}
    .mic-container .stButton button .st-emotion-cache {{
        padding: 0;
    }}
    .mic-listening .stButton button {{
        animation: pulse 1.2s ease-in-out infinite;
        background: linear-gradient(135deg, #f093fb, #f5576c);
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(245,87,108,0.5); }}
        70% {{ box-shadow: 0 0 0 24px rgba(245,87,108,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(245,87,108,0); }}
    }}

    /* ── Status bar ────────────────────────────────── */
    .status-bar {{
        background: {theme_status_bg};
        border-radius: 40px;
        padding: 0.5rem 1.25rem;
        margin: 0.5rem 0 1rem 0;
        text-align: center;
        color: {theme_text_muted};
        font-size: 0.85rem;
        border: 1px solid {theme_border};
        min-height: 2.4rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }}
    .status-bar .spinner {{
        display: inline-block;
        width: 14px;
        height: 14px;
        border: 2px solid rgba(161,140,209,0.2);
        border-top-color: #a18cd1;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }}
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}

    /* ── Section headings ──────────────────────────── */
    .section-heading {{
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {theme_text_muted};
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid {theme_border};
    }}

    /* ── Chat cards ────────────────────────────────── */
    .chat-card {{
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin: 0.45rem 0;
        border: 1px solid {theme_border};
        line-height: 1.5;
    }}
    .chat-card.user {{
        background: {theme_card_user};
        margin-right: 1.5rem;
    }}
    .chat-card.assistant {{
        background: {theme_card_ai};
        margin-left: 1.5rem;
    }}
    .chat-card .label {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }}
    .chat-card.user .label  {{ color: #b9a4e0; }}
    .chat-card.assistant .label {{ color: #fbc2eb; }}
    .chat-card .content {{
        color: {theme_text};
        font-size: 0.92rem;
    }}

    /* ── Transcript box ────────────────────────────── */
    .transcript-box {{
        background: {theme_status_bg};
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        color: {theme_text};
        border: 1px solid {theme_border};
        font-family: 'Consolas', 'Courier New', monospace;
        margin: 0.25rem 0;
        white-space: pre-wrap;
        word-break: break-word;
    }}

    /* ── Settings sidebar tweaks ───────────────────── */
    .settings-label {{
        font-size: 0.75rem;
        color: {theme_text_muted};
        margin-bottom: 0.2rem;
    }}

    /* ── Audio player ──────────────────────────────── */
    .audio-wrapper {{
        margin-top: 0.2rem;
    }}
    .audio-wrapper audio {{
        width: 100%;
        height: 36px;
        border-radius: 6px;
    }}

    /* ── Hide Streamlit junk ───────────────────────── */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .stDeployButton {{ display: none !important; }}

    /* ── Sidebar ───────────────────────────────────── */
    section[data-testid="stSidebar"] .stApp {{
        background: {theme_status_bg};
    }}
    section[data-testid="stSidebar"] .stApp h2 {{
        font-size: 1.1rem;
    }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

def _on_wake() -> None:
    st.session_state.wake_triggered = True
    st.rerun()


def init_state() -> None:
    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = VoicePipeline()
            st.session_state.ready = True
            st.session_state.status = "Tap the mic or say 'Hey Nova'."
        except Exception as exc:
            st.session_state.ready = False
            st.session_state.status = f"Init failed: {exc}"

    for k in ("audio_path", "last_transcript", "last_response", "wake_triggered"):
        if k not in st.session_state:
            st.session_state[k] = None

    if "wake_triggered" not in st.session_state or st.session_state.wake_triggered is None:
        st.session_state.wake_triggered = False
    if "running" not in st.session_state:
        st.session_state.running = False
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    if "wake_detector" not in st.session_state:
        try:
            detector = WakeWordDetector(
                wake_word="hey nova",
                on_wake=_on_wake,
            )
            detector.start()
            st.session_state.wake_detector = detector
            if st.session_state.get("ready", False):
                st.session_state.status = "Listening for 'Hey Nova'..."
        except Exception as exc:
            st.session_state.wake_detector = None
            logging.getLogger(__name__).warning("Could not start wake word: %s", exc)


init_state()

# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------

def on_talk() -> None:
    detector = st.session_state.get("wake_detector")
    if detector and detector.is_running:
        detector.pause()
        time.sleep(0.5)  # let current listen cycle finish

    st.session_state.running = True
    st.session_state.status = "Listening..."
    st.session_state.last_transcript = None
    st.session_state.last_response = None
    st.session_state.audio_path = None

    pipeline: VoicePipeline = st.session_state.pipeline
    result: TurnResult = pipeline.run_once()

    if result.transcript:
        st.session_state.last_transcript = result.transcript
    if result.response:
        st.session_state.last_response = result.response
    if result.audio_path:
        st.session_state.audio_path = result.audio_path

    if result.success:
        st.session_state.status = "Done."
    else:
        st.session_state.status = result.error or "Something went wrong."
    st.session_state.running = False

    if detector and detector.is_running:
        detector.resume()


def on_stop() -> None:
    st.session_state.pipeline.stop_tts()
    st.session_state.status = "Stopped."


def on_clear() -> None:
    st.session_state.pipeline.clear_memory()
    st.session_state.last_transcript = None
    st.session_state.last_response = None
    st.session_state.audio_path = None
    st.session_state.status = "Memory cleared."


def on_toggle_history() -> None:
    st.session_state.show_history = not st.session_state.show_history


# -------------------------------------------------------------------
# Sidebar — Settings
# -------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    dark = st.toggle(
        "Dark Mode",
        value=st.session_state.dark_mode,
        key="dark_toggle",
        help="Switch between light and dark theme",
    )
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.markdown("---")

    st.markdown('<div class="settings-label">Microphone</div>',
                unsafe_allow_html=True)
    mic_index = st.number_input(
        "Device index",
        min_value=0,
        max_value=20,
        value=st.session_state.pipeline.device_index or 0,
        step=1,
        label_visibility="collapsed",
        key="mic_index",
    )

    st.markdown('<div class="settings-label">Wake Word</div>',
                unsafe_allow_html=True)
    wake_enabled = st.toggle(
        "Enable",
        value=st.session_state.get("wake_detector") is not None
               and st.session_state.wake_detector.is_running,
        key="wake_toggle",
        help="Listen for 'Hey Nova' to activate",
    )

    st.markdown("---")

    tone = st.select_slider(
        "Response tone",
        options=["Concise", "Balanced", "Detailed"],
        value="Balanced",
        key="tone",
    )

    st.markdown("---")
    disabled = not st.session_state.get("ready", False)
    st.button("🧹 Clear memory", on_click=on_clear,
              disabled=disabled, type="secondary", use_container_width=True)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.markdown(
    '<div class="nova-header"><h1>Nova</h1>'
    '<p>AI Voice Agent</p></div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# 🎤 Mic button
# -------------------------------------------------------------------

running = st.session_state.running
disabled = not st.session_state.get("ready", False) or running

st.markdown(
    f'<div class="mic-container{" mic-listening" if running else ""}>',
    unsafe_allow_html=True,
)
action = on_stop if running else on_talk
st.button("⏹" if running else "🎤", on_click=action,
          disabled=disabled, type="primary", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
label = "Listening..." if running else "Tap to talk"
st.markdown(
    f'<p style="text-align:center;color:{theme_text_muted};'
    f'font-size:0.8rem;margin:-0.4rem 0 0.6rem 0;">{label}</p>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Status
# -------------------------------------------------------------------

status_text = st.session_state.status or ""
spinner = '<span class="spinner"></span>' if running else ""
st.markdown(
    f'<div class="status-bar">{spinner} {status_text}</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Wake-word trigger
# -------------------------------------------------------------------

wake_triggered = st.session_state.get("wake_triggered", False)
if wake_triggered and not running and st.session_state.get("ready", False):
    st.session_state.wake_triggered = False
    on_talk()
    st.rerun()

# -------------------------------------------------------------------
# Conversation
# -------------------------------------------------------------------

st.markdown('<div class="section-heading">💬 Conversation</div>',
            unsafe_allow_html=True)

if st.session_state.last_transcript:
    st.markdown(
        f'<div class="chat-card user">'
        f'<div class="label">You</div>'
        f'<div class="content">{st.session_state.last_transcript}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

if st.session_state.last_response:
    st.markdown(
        f'<div class="chat-card assistant">'
        f'<div class="label">Nova</div>'
        f'<div class="content">{st.session_state.last_response}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

if not st.session_state.last_transcript and not st.session_state.last_response:
    st.markdown(
        f'<div style="text-align:center;color:{theme_text_muted};'
        f'font-size:0.85rem;padding:2rem 0;">'
        f'Press 🎤 and ask something...</div>',
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Transcript (raw text)
# -------------------------------------------------------------------

if st.session_state.last_transcript:
    st.markdown('<div class="section-heading">📝 Transcript</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="transcript-box">{st.session_state.last_transcript}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.audio_path:
        ap = st.session_state.audio_path
        if Path(ap).is_file():
            with open(ap, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")

# -------------------------------------------------------------------
# History
# -------------------------------------------------------------------

if st.session_state.get("ready"):
    history = st.session_state.pipeline.memory.get_history()
    if history:
        cols = st.columns([6, 1])
        with cols[0]:
            st.markdown(
                f'<div class="section-heading">📜 History ({len(history)//2} turns)</div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.button(
                "🔄" if not st.session_state.show_history else "▲",
                on_click=on_toggle_history,
                help="Toggle history visibility",
                key="hist_toggle",
            )

        if st.session_state.show_history:
            for msg in history:
                role = msg["role"]
                label = "You" if role == "user" else "Nova"
                st.markdown(
                    f'<div class="chat-card {role}">'
                    f'<div class="label">{label}</div>'
                    f'<div class="content">{msg["content"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
