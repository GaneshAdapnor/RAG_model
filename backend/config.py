"""Configuration settings for the RAG system."""
import os
from pathlib import Path
from typing import Literal

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try loading from current directory
        load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()  # "openrouter", "groq", "openai", or "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# OpenRouter Configuration (primary provider)
# OPENROUTER_API_KEYS is a comma-separated list used for automatic failover
# when a key hits a rate limit. OPENROUTER_API_KEY (singular) is also
# honored as a convenience for a single-key setup.
_openrouter_keys_raw = os.getenv("OPENROUTER_API_KEYS", "") or os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_KEYS = [k.strip() for k in _openrouter_keys_raw.split(",") if k.strip()]
OPENROUTER_API_KEY = OPENROUTER_API_KEYS[0] if OPENROUTER_API_KEYS else ""
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Groq Configuration (secondary/fallback provider, still supported)
# GROQ_API_KEYS is a comma-separated list used for automatic failover when a
# key hits a rate limit. GROQ_API_KEY (singular) is also honored as a
# convenience for a single-key setup.
_groq_keys_raw = os.getenv("GROQ_API_KEYS", "") or os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in _groq_keys_raw.split(",") if k.strip()]
GROQ_API_KEY = GROQ_API_KEYS[0] if GROQ_API_KEYS else ""
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Ollama Embedding Configuration (for local models)
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Embeddings are always local/free (SentenceTransformers) regardless of which
# LLM provider is selected, since Groq has no embeddings endpoint.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Upload limits
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))

# Storage Configuration
# Each user session gets its own subdirectory under STORAGE_ROOT so that
# documents/vectors from one visitor are never visible to another
# (see VectorStoreManager, which appends "sessions/<session_id>/...").
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "./backend/storage"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./backend/storage/faiss_index")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./backend/storage/uploads")

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "5"))

# Summarization Configuration
SUMMARY_MAX_LENGTH = int(os.getenv("SUMMARY_MAX_LENGTH", "500"))
AUTO_SUMMARIZE = os.getenv("AUTO_SUMMARIZE", "true").lower() == "true"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

