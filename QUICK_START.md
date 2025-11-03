# Quick Start Guide - RAG PDF Q&A System

## Step 1: Install Dependencies

```bash
pip install -r requirements_rag.txt
```

**Important**: Install at least one PDF processing library:
```bash
pip install PyMuPDF  # Recommended
# OR
pip install pdfplumber
# OR
pip install PyPDF2
```

## Step 2: Set OpenAI API Key (Optional)

**Option A**: Set environment variable
```bash
# Windows
set OPENAI_API_KEY=your_key_here

# Linux/Mac
export OPENAI_API_KEY=your_key_here
```

**Option B**: Enter in Streamlit sidebar (when app runs)

**Note**: If no API key, the system will use SentenceTransformers (free, but slower)

## Step 3: Run the Application

```bash
streamlit run rag_streamlit.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Step 4: Use the Application

1. **Upload PDFs**: Click "Browse files" and select your PDF files
2. **Process**: Click "🔄 Process Documents"
   - Wait for processing to complete
   - You'll see progress bars and success messages
3. **Ask Questions**: Enter a question and click "🔍 Get Answer"
   - Get instant answers with source citations
   - View page numbers and file sources

## Features Overview

✅ **Multiple PDF Support**: Upload several PDFs at once
✅ **Smart Caching**: First processing creates cache, next time is instant
✅ **Source Attribution**: See exactly which file and page the answer came from
✅ **Processing Indicators**: Visual feedback during all operations
✅ **Clear Options**: Clear documents or cache anytime

## Troubleshooting

**"No PDF extraction library available"**
→ Install: `pip install PyMuPDF`

**"No embeddings model available"**
→ Install: `pip install sentence-transformers`
→ OR set OPENAI_API_KEY

**Slow processing first time**
→ Normal! First run downloads models and creates cache
→ Subsequent runs are much faster

## Example Questions

- "What is the main topic of this document?"
- "What are the key findings?"
- "Summarize the document"
- "What methodology was used?"
- "What are the conclusions?"

Enjoy using the RAG system! 🚀

