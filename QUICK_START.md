# Quick Start Guide - RAG Document Q&A System

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set your OpenRouter API Key

Get a free key from [OpenRouter Console](https://openrouter.ai/settings/keys).

**Option A**: Create a `.env` file (copy `.env.example`)
```bash
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=your-key-here
```

**Option B**: Set an environment variable
```bash
# Windows
set OPENROUTER_API_KEY=your_key_here

# Linux/Mac
export OPENROUTER_API_KEY=your_key_here
```

The key is read once at startup from your environment/secrets — there's no in-app field for it, since it's a deployer-configured setting, not something each visitor manages.

## Step 3: Run the Application

```bash
streamlit run rag_streamlit.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Step 4: Use the Application

1. **Upload documents**: PDF, Word (.docx), text, Markdown, HTML, or CSV
2. **Process**: Click "🔄 Process documents" in the Upload tab
   - Each file gets chunked, embedded, and (optionally) auto-summarized
3. **Chat**: Switch to the Chat tab and ask questions
   - Answers include source chunks so you can verify them
   - Chat history is kept for the session

## Features Overview

✅ **Multi-format documents**: PDF, Word, TXT, Markdown, HTML, CSV, RTF
✅ **Chat-style Q&A** with persistent history and source citations
✅ **Automatic key failover**: add multiple OpenRouter keys and the app rotates past rate limits
✅ **Per-session isolation**: your documents are private to your browser session
✅ **Auto-summarization** with key points on upload

## Troubleshooting

**"No OpenRouter API key configured"**
→ Add one via `.env`, an environment variable, or the sidebar

**"All configured OpenRouter API keys are currently rate-limited"**
→ Wait a minute, or add another free key from the OpenRouter Console

**"File exceeds the X MB limit"**
→ Split the document or raise `MAX_FILE_SIZE_MB` in `.env`

**Slow first run**
→ Normal — the embeddings model downloads once and is cached afterward

## Example Questions

- "What is the main topic of this document?"
- "What are the key findings?"
- "Summarize the document"
- "What methodology was used?"
- "What are the conclusions?"

Enjoy using the RAG system! 🚀
