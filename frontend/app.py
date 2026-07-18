"""HR Assistant — Streamlit frontend.

A premium chat UI over the RAG backend. Structured in clear sections:
config & assets → styling → backend calls → state → sidebar → main view.
"""

import base64
import json
import os
import uuid

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSISTANT_NAME = "Anchor"
REQUEST_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Assets — inline SVG avatars (no external requests, crisp on any screen).
# Defined before set_page_config so the robot mark can serve as the favicon.
# ---------------------------------------------------------------------------
def _svg(markup: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(markup.encode()).decode()


# Robot mark — drawn in white on a 48×48 canvas, nudged down to sit centered.
# Reused for the brand mark, chat avatar, hero logo, and favicon so branding
# is cohesive.
_ROBOT_BODY = (
    "<g transform='translate(0,3)' fill='none' stroke='#fff'"
    " stroke-width='3.6' stroke-linecap='round' stroke-linejoin='round'>"
    "<circle cx='24' cy='7' r='3.2' fill='#fff' stroke='none'/>"
    "<path d='M24 10.4 V15'/>"
    "<rect x='10.5' y='15' width='27' height='21.5' rx='7'/>"
    "<path d='M7.5 22 V28 M40.5 22 V28'/>"
    "<circle cx='19' cy='24.5' r='2.7' fill='#fff' stroke='none'/>"
    "<circle cx='29' cy='24.5' r='2.7' fill='#fff' stroke='none'/>"
    "<path d='M18 31 H30'/>"
    "</g>"
)


def _robot_svg(with_gradient_tile: bool) -> str:
    tile = ""
    if with_gradient_tile:
        tile = (
            "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
            "<stop offset='0' stop-color='#6366F1'/><stop offset='1' stop-color='#A855F7'/>"
            "</linearGradient></defs><rect width='48' height='48' rx='14' fill='url(#g)'/>"
        )
    return _svg(
        "<svg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 48 48'>"
        f"{tile}{_ROBOT_BODY}</svg>"
    )


BOT_AVATAR = _robot_svg(with_gradient_tile=True)   # white robot on gradient tile
ROBOT_WHITE = _robot_svg(with_gradient_tile=False)  # transparent, for the gradient brand mark
USER_AVATAR = _svg(
    "<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40' viewBox='0 0 40 40'>"
    "<rect width='40' height='40' rx='12' fill='#1a1d27'/>"
    "<circle cx='20' cy='16' r='6' fill='#8B92A0'/>"
    "<path d='M8 33c0-6.5 5.4-9.5 12-9.5S32 26.5 32 33z' fill='#8B92A0'/></svg>"
)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call. Favicon = the robot mark.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Anchor — HR Assistant",
    page_icon=BOT_AVATAR,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Suggested prompts shown on the empty state: (icon, title, full question)
SUGGESTIONS = [
    ("🌴", "Leave policy", "How many earned leaves am I entitled to in a year?"),
    ("🏠", "Remote work", "What is the company's remote work policy?"),
    ("💰", "Payroll", "When are salaries paid and how is pay calculated?"),
    ("🤰", "Maternity", "What are the maternity leave benefits?"),
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def load_styles() -> None:
    with open(os.path.join(APP_DIR, "styles.css"), encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
@st.cache_data(ttl=8, show_spinner=False)
def backend_online() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def stream_backend(prompt: str, session_id: str):
    """Stream a reply from /query/stream (NDJSON).

    Yields text deltas; collects sources/errors on self. Falls back to a
    friendly error message as a single yielded chunk."""
    stream_backend.sources = []
    stream_backend.error = None
    try:
        resp = requests.post(
            f"{BACKEND_URL}/query/stream",
            json={"query": prompt, "session_id": session_id},
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        if resp.status_code == 429:
            yield "⚠️ You're sending messages a little too fast — give me a moment and try again."
            return
        if resp.status_code != 200:
            yield "⚠️ Something went wrong on the server. Please try again shortly."
            return
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "meta":
                stream_backend.sources = event.get("sources", [])
            elif event["type"] == "delta":
                yield event["text"]
            elif event["type"] == "error":
                yield f"⚠️ {event.get('message', 'Something went wrong.')}"
                return
    except requests.Timeout:
        yield "⏳ That took longer than expected. Please try again."
    except requests.RequestException:
        yield "🔌 I can't reach the backend right now. Make sure it's running and try again."


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_sources(sources: list) -> None:
    if not sources:
        return
    pages = sorted({s.get("page") for s in sources if s.get("page")})
    label = f"📄 {len(sources)} source passages"
    if pages:
        label += " · page " + ", ".join(str(p) for p in pages)
    with st.expander(label):
        for s in sources:
            snippet = " ".join(str(s.get("text", "")).split())[:340]
            page = s.get("page", "?")
            st.markdown(
                f"<div class='source-item'><span class='source-tag'>p. {page}</span>"
                f"<span>{snippet}…</span></div>",
                unsafe_allow_html=True,
            )


def render_message(msg: dict) -> None:
    role = msg["role"]
    avatar = USER_AVATAR if role == "user" else BOT_AVATAR
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        if role == "assistant":
            render_sources(msg.get("sources", []))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
def init_state() -> None:
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)


def start_new_chat() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.pending = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='brand'><div class='brand-mark'>"
            f"<img src='{ROBOT_WHITE}' width='26' height='26' alt='Anchor'/></div>"
            "<div><div class='brand-name'>Anchor</div>"
            "<div class='brand-sub'>HR Intelligence</div></div></div>",
            unsafe_allow_html=True,
        )

        st.button("✧  New chat", key="new_chat", type="primary", on_click=start_new_chat)

        online = backend_online()
        dot = "on" if online else "off"
        label = "Connected" if online else "Offline"
        st.markdown(
            f"<div class='side-label'>System status</div>"
            f"<div class='status-pill'><span class='status-dot {dot}'></span>{label}</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div class='side-label'>Popular topics</div>", unsafe_allow_html=True)
        st.markdown(
            "".join(
                f"<div class='topic'><span class='dot'></span>{t}</div>"
                for t in ["Leave & attendance", "Payroll & wages", "Remote work", "Maternity benefits", "Code of conduct"]
            ),
            unsafe_allow_html=True,
        )

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown(
            "<div class='side-foot'>Grounded in your <b>HR policy</b> documents.<br/>"
            "Crafted by <b>Likith</b></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------
def render_hero() -> None:
    st.markdown(
        "<div class='hero'>"
        f"<img class='hero-logo' src='{BOT_AVATAR}' width='68' height='68' alt='Anchor'/>"
        "<div class='hero-badge'>✨ Powered by Retrieval-Augmented AI</div>"
        "<h1>Meet <span class='grad'>Anchor</span>,<br/>your HR assistant</h1>"
        "<p>Ask anything about policies, leave, payroll, benefits, or compliance — "
        "answered straight from your company handbook.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='suggest-label'>Try asking</div>", unsafe_allow_html=True)

    st.markdown("<div class='hero-suggest'>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, title, question) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(f"{icon}  {title}", key=f"sug_{i}"):
            st.session_state.pending = question
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def handle_turn(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        answer = st.write_stream(stream_backend(prompt, st.session_state.session_id))
        sources = stream_backend.sources
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main() -> None:
    load_styles()
    init_state()
    render_sidebar()

    if st.session_state.messages:
        for msg in st.session_state.messages:
            render_message(msg)
    else:
        render_hero()

    # A prompt can arrive from the composer or from a clicked suggestion.
    prompt = st.chat_input("Message Anchor…")
    if st.session_state.pending:
        prompt = st.session_state.pending
        st.session_state.pending = None

    if prompt:
        handle_turn(prompt)


main()
