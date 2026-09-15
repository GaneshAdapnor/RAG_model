# 📚 RAG Document Q&A

Upload a document, get grounded answers with citations — no hallucinated facts, no manual searching through PDFs.

A Retrieval-Augmented Generation system built end-to-end: multi-format document ingestion, local embeddings, FAISS vector search, and LLM answering via OpenRouter — wrapped in a polished, chat-style Streamlit UI with a real dark/light mode.

**🔗 Live demo:** [rag-document-model.streamlit.app](https://rag-document-model.streamlit.app/)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-vector%20search-informational)](https://github.com/facebookresearch/faiss)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-multi--model%20LLM-8b5cf6)](https://openrouter.ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What it does

- **Upload** PDF, Word, plain text, Markdown, HTML, CSV, or RTF files — no format lock-in
- **Ask questions** in a chat interface and get answers grounded in your documents, with the exact source chunks shown alongside each answer
- **Get instant summaries and key points** for every document on upload
- **Never lose your place** — chat history and uploaded documents persist for your session, invisible to anyone else's
- **Switch themes instantly** — a hand-built neumorphic light/dark mode that never round-trips to the server

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Fast to build, easy to deploy, good enough for a real chat-style product |
| LLM | OpenRouter (model-agnostic) | One API key, access to many models (DeepSeek, Llama, Claude, GPT, ...); supports automatic multi-key failover |
| Embeddings | SentenceTransformers (local) | Free, no external dependency, and decoupled from whichever LLM provider is active |
| Vector store | FAISS | Lightweight, in-process, no external database to run |
| Orchestration | LangChain (LCEL) | Chunking, retrieval, prompt composition |
| API (optional) | FastAPI | The same core logic is exposed as a standalone REST API for non-Streamlit clients |

## Architecture

```
┌─────────────────────┐
│   rag_streamlit.py   │   Chat UI, upload flow, theming
└──────────┬───────────┘
           │ imports directly (no HTTP hop)
┌──────────▼───────────┐
│      backend/         │   Shared core library
│  ├─ document_processor │   PDF/DOCX/TXT/MD/HTML/CSV/RTF → text
│  ├─ vector_store       │   FAISS index + metadata, per-session
│  ├─ llm_service        │   OpenRouter/Groq/OpenAI/Ollama abstraction + key failover
│  ├─ rag_chain          │   Retrieval + prompt + answer (LCEL)
│  └─ summarization      │   Summary + key points
└──────────┬───────────┘
           │ also mountable as a standalone API
┌──────────▼───────────┐
│      backend/app.py    │   FastAPI wrapper (uvicorn backend.app:app)
└───────────────────────┘
```

The Streamlit app and the FastAPI service share the exact same `backend/` core — no duplicated logic between the two entry points.

## Engineering decisions worth noting

- **Per-session isolation**: every visitor gets their own FAISS index and metadata file (`backend/storage/sessions/<session_id>/`), so one user's uploaded documents are never retrievable by another. This closes a real privacy gap that exists in a lot of demo RAG apps, where all uploads land in one shared index.
- **Provider-agnostic LLM layer with key failover**: `LLMService` treats the model provider as swappable configuration, not a hardcoded dependency — OpenRouter is the default, Groq/OpenAI/Ollama are drop-in alternatives. Any provider can be configured with multiple comma-separated API keys; on a rate-limit error, the service automatically rotates to the next key before ever surfacing an error to the user.
- **Local-first embeddings**: embeddings never depend on which LLM provider is active — they're always local SentenceTransformers, which keeps the app free to run and avoids an extra network dependency on the hot path.
- **Multi-format ingestion**: one `DocumentProcessor` handles PDF, DOCX, TXT, MD, HTML, CSV, and RTF so the app isn't limited to a single file type.
- **Zero-latency theme switching**: the light/dark toggle is pure CSS/HTML — a hidden checkbox read by a `:has()` selector that flips CSS custom properties for the entire page. No Streamlit widget, no server rerun, no flash of the wrong theme. (An earlier version used a Python-driven toggle; it was replaced after hitting real state-desync bugs, which is a good illustration of picking the right tool for a purely client-side concern.)
- **Dual entry point, one core**: the FastAPI backend isn't a leftover — it's kept in sync with the Streamlit app because both import the same `backend/*` modules, so the project is usable as an API by other clients without maintaining two implementations.

## Getting started

### Prerequisites
- Python 3.9+
- A free [OpenRouter API key](https://openrouter.ai/settings/keys)

### Run locally

```bash
git clone https://github.com/GaneshAdapnor/RAG_model.git
cd RAG_model
pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=your-key-here

streamlit run rag_streamlit.py
```

Open `http://localhost:8501`, upload a document, and start asking questions.

### Deploy to Streamlit Cloud

See [DEPLOY_TO_STREAMLIT.md](DEPLOY_TO_STREAMLIT.md) for the full walkthrough. In short: point Streamlit Cloud at `rag_streamlit.py`, and add `OPENROUTER_API_KEYS` (or `OPENROUTER_API_KEY`) under **Advanced settings → Secrets**.

### Run the standalone API (optional)

```bash
uvicorn backend.app:app --reload
```

API docs at `http://localhost:8000/docs`. Endpoints: `POST /api/upload`, `POST /api/query`, `POST /api/summarize`, `GET /api/documents`, `DELETE /api/documents/{doc_id}`. Pass an `X-Session-Id` header to keep documents scoped to a particular caller.

## Configuration

All configuration is environment-driven — see [`.env.example`](.env.example) for the full list. The essentials:

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEYS=key-one,key-two    # comma-separated for automatic failover
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
AUTO_SUMMARIZE=true
```

## Project structure

```
RAG_model/
├── rag_streamlit.py         # Streamlit UI (chat, upload, theming, document manager)
├── backend/
│   ├── app.py                # FastAPI app (standalone API)
│   ├── config.py              # Environment-driven configuration
│   ├── document_processor.py  # Multi-format text extraction
│   ├── vector_store.py        # FAISS persistence, per-session
│   ├── llm_service.py         # OpenRouter/Groq/OpenAI/Ollama provider abstraction
│   ├── rag_chain.py           # Retrieval-augmented answering (LCEL)
│   ├── summarization.py       # Summary + key point extraction
│   └── models.py              # Pydantic request/response schemas
├── requirements.txt
└── .env.example
```

## License

MIT
