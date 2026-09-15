"""
Retrieval-Augmented Generation (RAG) System — Streamlit UI

Upload documents (PDF, Word, text, Markdown, HTML, CSV, RTF), get automatic
summaries, and chat with your documents. Answering is powered by a
background AI API (OpenRouter) plus FAISS for retrieval and local
SentenceTransformers embeddings.

This file is a thin UI layer over backend/*, which contains all the actual
document processing, vector store, and RAG logic.
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path

# Load environment variables from .env file (local dev convenience)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

# IMPORTANT: st.set_page_config must be the FIRST Streamlit command.
try:
    st.set_page_config(
        page_title="RAG Document Q&A",
        page_icon="📚",
        layout="wide",
    )
except Exception:
    pass

# Make sure the repo root is importable so `backend.*` resolves regardless
# of the working directory Streamlit was launched from.
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import MAX_FILE_SIZE_MB, AUTO_SUMMARIZE, LLM_PROVIDER
from backend.document_processor import DocumentProcessor
from backend.vector_store import VectorStoreManager
from backend.llm_service import LLMService, OPENAI_COMPATIBLE_AVAILABLE, NoAPIKeysError, AllAPIKeysExhaustedError
from backend.summarization import SummarizationService
from backend.rag_chain import RAGChain

# Which env var prefix to read API keys from, per provider. The active
# provider is set via LLM_PROVIDER in backend/config.py (defaults to
# "openrouter"). No model or provider choice is ever exposed in the UI —
# it's a background implementation detail configured by the deployer.
_PROVIDER_ENV_PREFIX = {
    "openrouter": "OPENROUTER",
    "groq": "GROQ",
}

SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "md", "html", "htm", "csv", "rtf"]


# ---------------------------------------------------------------------------
# Session state / service wiring
# ---------------------------------------------------------------------------

def get_api_keys_from_env_or_secrets() -> list:
    """Read the active provider's API keys, configured by the app owner via
    Streamlit secrets or env vars (the recommended path for a deployed app)."""
    prefix = _PROVIDER_ENV_PREFIX.get(LLM_PROVIDER, LLM_PROVIDER.upper())
    keys_name, key_name = f"{prefix}_API_KEYS", f"{prefix}_API_KEY"
    raw = ""
    try:
        raw = st.secrets.get(keys_name, "") or st.secrets.get(key_name, "")
    except Exception:
        pass
    if not raw:
        raw = os.getenv(keys_name, "") or os.getenv(key_name, "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = get_api_keys_from_env_or_secrets()


# ---------------------------------------------------------------------------
# Theming — neumorphic (soft UI): surfaces share the page's base color and
# read as raised or inset purely through a pair of soft shadows, no visible
# borders. The light/dark switch is pure CSS/HTML — a checkbox whose
# :checked state is read by a `:has()` selector on the app's root container,
# which then overrides the CSS custom properties for the whole subtree.
# There is deliberately NO Streamlit widget, session_state, or rerun
# involved in the switch itself: earlier attempts using st.toggle (with or
# without st.fragment) kept desyncing other widgets (the file uploader) or
# leaving stale/duplicate <style> blocks behind, because every one of those
# depended on a script rerun actually re-executing the right code at the
# right time. A pure client-side toggle has no such race to get wrong.
# ---------------------------------------------------------------------------

_THEME_CSS = """
<style>
:root {
    --app-bg: #e6e7ee;
    --app-text: #2c2c3a;
    --app-muted: #6b6b7b;
    --app-accent: #6C5CE7;
    --app-accent-text: #ffffff;
    --shadow-light: #ffffff;
    --shadow-dark: #b8bac2;
}
[data-testid="stApp"]:has(#theme-toggle-checkbox:checked) {
    --app-bg: #2b2f3a;
    --app-text: #eceaf5;
    --app-muted: #9a9bb0;
    --app-accent: #8b7cf6;
    --app-accent-text: #12111a;
    --shadow-light: #363b48;
    --shadow-dark: #1e212a;
}

[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"] {
    background-color: var(--app-bg) !important;
}
/* Color only actual text-bearing tags, not every descendant — forcing
   color on `*` bled into Streamlit's internal file-chip icon (an
   svg/masked element) and broke its rendering. */
[data-testid="stAppViewContainer"] :is(p, span, label, li, small, strong, em, code, a, h1, h2, h3, h4, h5, h6, td, th),
[data-testid="stSidebar"] :is(p, span, label, li, small, strong, em, code, a, h1, h2, h3, h4, h5, h6, td, th),
[data-testid="stBottom"] :is(p, span, label, li, small, strong, em, code, a, h1, h2, h3, h4, h5, h6, td, th) {
    color: var(--app-text);
}
/* The uploaded-file row's filename has no stable data-testid in this
   Streamlit version — target it structurally within the dropzone instead
   so it stays legible in dark mode. */
[data-testid="stFileUploaderDropzone"] ~ div span,
[data-testid="stFileUploaderDropzone"] ~ div small {
    color: var(--app-text) !important;
}

h1, h2, h3 {
    font-weight: 700;
    letter-spacing: -0.01em;
}

[data-testid="stCaptionContainer"], .stCaption, small {
    color: var(--app-muted) !important;
}

hr, [data-testid="stDivider"] {
    border: none !important;
    border-top: 1px solid var(--shadow-dark) !important;
}

/* Neumorphic raised surfaces: file uploader, expanders, chat bubbles, status widgets */
[data-testid="stFileUploaderDropzone"],
[data-testid="stExpander"],
[data-testid="stChatMessage"],
[data-testid="stStatusWidget"],
[data-testid="stChatInput"],
div[data-baseweb="notification"] {
    background-color: var(--app-bg) !important;
    border: none !important;
    border-radius: 16px !important;
    box-shadow: 6px 6px 12px var(--shadow-dark), -6px -6px 12px var(--shadow-light);
}

/* Expander header: Streamlit applies its own hover highlight (a light
   overlay) that isn't theme-aware, washing out text in dark mode. Force
   it transparent so only our own themed background ever shows through. */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus,
[data-testid="stExpander"] [role="button"],
[data-testid="stExpander"] [role="button"]:hover,
[data-testid="stExpander"] [role="button"]:focus {
    background-color: transparent !important;
    color: var(--app-text) !important;
}
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] [role="button"] svg {
    fill: var(--app-text) !important;
}

/* The actual editable elements (textarea/input) — and every wrapper div
   BaseWeb nests them inside — keep their own light/translucent fill by
   default, which doesn't match our theme background. Force every layer,
   not just the textarea itself, since BaseWeb wraps inputs in 2-3 nested
   divs that each can carry their own background. */
[data-testid="stChatInputTextArea"],
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] div,
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] div {
    background-color: var(--app-bg) !important;
    color: var(--app-text) !important;
    -webkit-text-fill-color: var(--app-text) !important;
    caret-color: var(--app-text) !important;
}
[data-testid="stChatInputTextArea"]::placeholder,
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: var(--app-muted) !important;
    opacity: 1 !important;
}

/* Buttons: raised by default, inset ("pressed") on click */
.stButton > button, button[kind="secondary"] {
    background-color: var(--app-bg) !important;
    color: var(--app-text) !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 4px 4px 8px var(--shadow-dark), -4px -4px 8px var(--shadow-light);
    transition: box-shadow 0.12s ease;
}
.stButton > button:active {
    box-shadow: inset 3px 3px 6px var(--shadow-dark), inset -3px -3px 6px var(--shadow-light) !important;
}
button[kind="primary"]:not(:disabled), .stButton > button[kind="primary"]:not(:disabled) {
    background-color: var(--app-accent) !important;
    color: var(--app-accent-text) !important;
    font-weight: 600 !important;
}
.stButton > button:disabled {
    opacity: 0.45 !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
}

/* Tabs */
[data-testid="stTabs"] {
    box-shadow: inset 3px 3px 6px var(--shadow-dark), inset -3px -3px 6px var(--shadow-light);
    border-radius: 14px;
    padding: 4px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background-color: var(--app-bg) !important;
    box-shadow: 4px 4px 8px var(--shadow-dark), -4px -4px 8px var(--shadow-light) !important;
    border-radius: 10px !important;
    color: var(--app-accent) !important;
}

/* Text inputs: subtle inset */
[data-testid="stTextInput"] > div {
    box-shadow: inset 3px 3px 6px var(--shadow-dark), inset -3px -3px 6px var(--shadow-light) !important;
    border-radius: 12px !important;
}

/* Hide the chat form's submit button — pressing Enter in the text input
   still submits the form even though the button itself is invisible. */
[data-testid="stForm"] [data-testid="stFormSubmitButton"] {
    display: none !important;
}

/* Pure CSS/HTML light/dark switch — no Streamlit widget involved */
.theme-switch {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    user-select: none;
    padding: 4px 0 12px 0;
}
.theme-switch input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}
.theme-switch-track {
    position: relative;
    width: 44px;
    height: 24px;
    border-radius: 999px;
    background: var(--app-bg);
    box-shadow: inset 3px 3px 6px var(--shadow-dark), inset -3px -3px 6px var(--shadow-light);
    flex-shrink: 0;
}
.theme-switch-track::before {
    content: "";
    position: absolute;
    top: 3px;
    left: 3px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--app-accent);
    transition: transform 0.2s ease;
}
.theme-switch input:checked ~ .theme-switch-track::before {
    transform: translateX(20px);
}
.theme-switch-label {
    color: var(--app-text);
    font-size: 0.95rem;
}
.theme-switch-label-off { display: inline; }
.theme-switch-label-on { display: none; }
.theme-switch input:checked ~ .theme-switch-label .theme-switch-label-off { display: none; }
.theme-switch input:checked ~ .theme-switch-label .theme-switch-label-on { display: inline; }
</style>
"""

_THEME_TOGGLE_HTML = """
<label class="theme-switch">
    <input type="checkbox" id="theme-toggle-checkbox">
    <span class="theme-switch-track"></span>
    <span class="theme-switch-label">
        <span class="theme-switch-label-off">☀️ Light mode</span>
        <span class="theme-switch-label-on">🌙 Dark mode</span>
    </span>
</label>
"""


def inject_theme_css():
    """Injected exactly once. Both palettes live in this single static
    stylesheet — switching them is handled entirely by the browser via the
    checkbox's :checked state, never by re-running Python or Streamlit.
    """
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def render_theme_toggle():
    st.markdown(_THEME_TOGGLE_HTML, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_vector_store_manager(session_id: str) -> VectorStoreManager:
    """One VectorStoreManager per browser session, kept alive across
    reruns so the FAISS index isn't reloaded from disk on every interaction.
    Cached by session_id, so different visitors never share documents.
    """
    manager = VectorStoreManager(session_id=session_id)
    manager.initialize()
    return manager


def friendly_error_message(exc: Exception) -> str:
    if isinstance(exc, NoAPIKeysError):
        return "🔑 No API key configured. Set the provider's API key in the app's environment or secrets."
    if isinstance(exc, AllAPIKeysExhaustedError):
        return "🐢 All configured API keys are currently rate-limited. Please wait a minute and try again."
    text = str(exc)
    if "401" in text or "invalid_api_key" in text.lower() or "unauthorized" in text.lower():
        return "❌ Invalid API key. Please check the configured key(s)."
    if "429" in text or "rate" in text.lower():
        return "❌ API rate limit hit. Try again shortly, or configure another key for automatic failover."
    return f"❌ {text[:300]}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(manager: VectorStoreManager):
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        render_theme_toggle()

        # API access is configured by the deployer (env vars / Streamlit
        # secrets) — this is a status indicator only, not something a
        # visitor needs to manage.
        if st.session_state.api_keys:
            st.caption(f"🟢 AI connected · {len(st.session_state.api_keys)} key(s)")
        else:
            prefix = _PROVIDER_ENV_PREFIX.get(LLM_PROVIDER, LLM_PROVIDER.upper())
            st.caption(f"🔴 AI not configured — set {prefix}_API_KEY in secrets")

        st.divider()
        st.subheader("📄 Your Documents")
        docs = manager.get_documents_metadata()
        if not docs:
            st.caption("No documents uploaded yet.")
        else:
            for doc_id, meta in docs.items():
                name_col, delete_col = st.columns([4, 1])
                with name_col:
                    st.markdown(f"📎 {meta.get('filename', 'unknown')}")
                with delete_col:
                    if st.button("🗑️", key=f"del_{doc_id}", help="Delete this document"):
                        manager.remove_document(doc_id)
                        st.rerun()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🗑️ Clear docs", use_container_width=True):
                manager.clear_all()
                st.rerun()


# ---------------------------------------------------------------------------
# Document upload tab
# ---------------------------------------------------------------------------

def render_upload_tab(manager: VectorStoreManager):
    st.subheader("Upload documents")
    st.caption(
        f"Supported formats: {', '.join(SUPPORTED_FILE_TYPES).upper()} • "
        f"Max {MAX_FILE_SIZE_MB} MB per file"
    )

    auto_summarize = st.checkbox("Auto-summarize on upload", value=AUTO_SUMMARIZE)

    uploaded_files = st.file_uploader(
        "Choose files",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key="doc_uploader",
    )

    process_clicked = st.button(
        "🔄 Process documents", type="primary", disabled=not uploaded_files
    )

    if uploaded_files and process_clicked:
        if auto_summarize and st.session_state.api_keys:
            LLMService.set_api_keys(LLM_PROVIDER, st.session_state.api_keys)
        summarization_service = SummarizationService(manager)
        progress = st.progress(0.0)

        for idx, uploaded_file in enumerate(uploaded_files):
            with st.status(f"Processing {uploaded_file.name}...", expanded=False) as status:
                try:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    try:
                        text, file_type = DocumentProcessor.process_document(tmp_path, uploaded_file.name)
                    finally:
                        os.unlink(tmp_path)

                    doc_id = str(uuid.uuid4())
                    chunks_created = manager.add_document(
                        doc_id=doc_id,
                        text=text,
                        filename=uploaded_file.name,
                        file_type=file_type,
                    )
                    status.update(label=f"✅ {uploaded_file.name} — {chunks_created} chunks", state="complete")

                    if auto_summarize:
                        try:
                            summary, key_points = summarization_service.summarize_document(doc_id)
                            doc_meta = manager.get_document_metadata(doc_id)
                            if doc_meta:
                                doc_meta["summary"] = summary
                                doc_meta["key_points"] = key_points
                                manager._save_metadata()
                        except Exception as exc:
                            st.warning(f"Summarization skipped for {uploaded_file.name}: {friendly_error_message(exc)}")

                except ValueError as exc:
                    status.update(label=f"⚠️ {uploaded_file.name}: {exc}", state="error")
                except Exception as exc:
                    status.update(label=f"❌ {uploaded_file.name}: {friendly_error_message(exc)}", state="error")

            progress.progress((idx + 1) / len(uploaded_files))

        st.success("Done! Switch to the Chat tab to ask questions.")
        st.rerun()


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

def render_chat_tab(manager: VectorStoreManager):
    docs = manager.get_documents_metadata()
    if not docs:
        st.info("👆 Upload documents in the **Upload** tab first.")
        return

    # A plain widget (not st.chat_input, which Streamlit always pins to the
    # bottom of the viewport) so the question box stays fixed at the top of
    # this tab, with newest-first history rendered below it.
    with st.form(key="chat_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question",
            placeholder="Ask a question about your documents...",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask", type="primary")

    if asked and question:
        if not st.session_state.api_keys:
            st.error("🔑 AI isn't configured yet. Set the provider's API key in the app's environment or secrets.")
        else:
            with st.spinner("Thinking..."):
                try:
                    LLMService.set_api_keys(LLM_PROVIDER, st.session_state.api_keys)
                    rag_chain = RAGChain(manager)
                    result = rag_chain.answer_question(question)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                    })
                except Exception as exc:
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": friendly_error_message(exc),
                        "sources": [],
                    })
            st.rerun()

    # Newest first — a stack, not a scrolling transcript.
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("sources"):
                with st.expander("📑 Sources"):
                    for source in turn["sources"]:
                        st.markdown(f"**{source['filename']}** (chunk {source['chunk_index']})")
                        st.caption(source["content"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not OPENAI_COMPATIBLE_AVAILABLE:
        st.error("❌ langchain-openai is not installed. Run: pip install langchain-openai")
        return

    init_session_state()
    inject_theme_css()
    manager = get_vector_store_manager(st.session_state.session_id)

    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.1rem;">
            <span style="font-size:2.2rem; line-height:1;">📚</span>
            <span style="font-size:2rem; font-weight:700; letter-spacing:-0.02em;">RAG Document Q&amp;A</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Upload documents and chat with them, grounded in your own files.")

    render_sidebar(manager)

    tab_upload, tab_chat = st.tabs(["📁 Upload & Manage", "💬 Chat"])
    with tab_upload:
        render_upload_tab(manager)
    with tab_chat:
        render_chat_tab(manager)

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: var(--app-muted);'>"
        "Built with Streamlit, LangChain, and FAISS"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        try:
            st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")
        except Exception:
            pass
        st.error("❌ Application Error")
        st.code(str(e))
        st.code(traceback.format_exc())
