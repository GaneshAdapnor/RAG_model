# RAG PDF Q&A System

A complete Retrieval-Augmented Generation (RAG) system built with Streamlit that allows users to upload PDF files and ask questions based on their contents.

## Features

- ✅ **Multiple PDF Upload**: Upload and process multiple PDF files simultaneously
- ✅ **Smart Text Extraction**: Uses PyMuPDF, pdfplumber, or PyPDF2 (with priority fallback)
- ✅ **Intelligent Chunking**: Splits documents into manageable 1000-character chunks with overlap
- ✅ **Embedding Caching**: Caches embeddings to avoid re-processing PDFs
- ✅ **Dual Embeddings Support**: OpenAI embeddings (text-embedding-3-large) or SentenceTransformers fallback
- ✅ **Source Attribution**: Shows source filename and page number for each retrieved chunk
- ✅ **Processing Indicators**: Shows spinner and progress bars during processing
- ✅ **Clear Documents**: Option to clear uploaded documents
- ✅ **Clean Interface**: User-friendly Streamlit interface

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_rag.txt
```

### 2. Install PDF Processing Library (at least one)

**Recommended (best performance):**
```bash
pip install PyMuPDF
```

**Alternative:**
```bash
pip install pdfplumber
```

**Fallback:**
```bash
pip install PyPDF2
```

### 3. Set OpenAI API Key (Optional)

If you want to use OpenAI embeddings and GPT models:

**Option A: Environment Variable**
```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

**Option B: `.env` file**
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

**Option C: Streamlit Interface**
You can also enter your API key directly in the Streamlit sidebar.

> **Note**: If no OpenAI API key is provided, the system will use SentenceTransformers embeddings (free, but runs locally and may be slower).

## Usage

### Run the Application

```bash
streamlit run rag_streamlit.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Step-by-Step Guide

1. **Upload PDFs**: Click "Browse files" and select one or more PDF files
2. **Process Documents**: Click "🔄 Process Documents" button
   - The system will extract text from all PDFs
   - Split documents into chunks
   - Create embeddings and store them in FAISS
   - Cache embeddings for future use
3. **Ask Questions**: Enter your question in the text area and click "🔍 Get Answer"
   - The system retrieves relevant chunks
   - Generates answer using GPT model
   - Shows answer with source documents

### Features Usage

#### Embedding Caching
- First time processing a PDF: Creates and saves embeddings
- Subsequent times: Loads cached embeddings (much faster!)
- Clear cache: Use "🗑️ Clear Cache" button in sidebar

#### Source Attribution
- Each answer shows source file and page number
- Click "View detailed source information" for full source previews
- See exactly where information came from

#### Configuration
- **LLM Model**: Select GPT model in sidebar (gpt-4o-mini recommended for cost)
- **API Key**: Enter OpenAI API key in sidebar if not set as environment variable

## Architecture

### Components

1. **PDF Text Extraction**
   - PyMuPDF (priority) - Fast and accurate
   - pdfplumber (fallback) - Good table extraction
   - PyPDF2 (last resort) - Basic extraction

2. **Text Chunking**
   - RecursiveCharacterTextSplitter
   - Chunk size: 1000 characters
   - Overlap: 200 characters (preserves context)

3. **Embeddings**
   - **OpenAI** (text-embedding-3-large): High quality, requires API key
   - **SentenceTransformers** (all-MiniLM-L6-v2): Free, local, good quality

4. **Vector Store**
   - FAISS: Fast similarity search
   - Persisted to disk for caching

5. **Retrieval**
   - Top-K retrieval (K=5 by default)
   - Semantic similarity search

6. **Answer Generation**
   - LangChain RetrievalQA chain
   - GPT-4/GPT-3.5 models
   - Context-aware responses

## Code Structure

```python
# Main components:
- extract_text_from_pdf()          # Extract text with page numbers
- split_documents()                # Split into chunks
- get_embeddings()                 # Get embedding model
- get_or_create_vectorstore()     # Create/load FAISS store
- get_answer()                     # RAG Q&A
- format_source_docs()             # Format sources for display
```

## Configuration

Edit these constants in `rag_streamlit.py`:

```python
CHUNK_SIZE = 1000        # Size of text chunks
CHUNK_OVERLAP = 200      # Overlap between chunks
TOP_K = 5                # Number of chunks to retrieve
CACHE_DIR = ".rag_cache" # Cache directory
```

## Troubleshooting

### Issue: No PDF extraction library available
**Solution**: Install at least one:
```bash
pip install PyMuPDF
# OR
pip install pdfplumber
# OR
pip install PyPDF2
```

### Issue: SentenceTransformers slow on first run
**Solution**: First run downloads the model (~80MB). Subsequent runs are faster. Use OpenAI embeddings for better performance.

### Issue: Cache not working
**Solution**: Ensure `.rag_cache` directory exists and has write permissions.

### Issue: OpenAI API errors
**Solution**: 
- Check API key is correct
- Verify you have API credits
- Check rate limits

## Performance Tips

1. **Use PyMuPDF** for fastest PDF processing
2. **Cache embeddings** - first processing is slower, subsequent uses are instant
3. **Use OpenAI embeddings** for better quality (faster than SentenceTransformers on CPU)
4. **Choose appropriate LLM** - gpt-4o-mini is fast and cost-effective

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- FAISS
- At least one PDF processing library
- OpenAI API key (optional, can use SentenceTransformers)

## License

MIT License

## Notes

- All processing is done locally (except OpenAI API calls)
- PDFs are not stored, only extracted text and embeddings
- Cache can be safely deleted to free up space
- First run with SentenceTransformers will download the model

