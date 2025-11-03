"""LLM service supporting both OpenAI and Ollama."""
import os
from typing import Optional
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings

try:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
    OLLAMA_AVAILABLE = True
    OLLAMA_EMBEDDINGS_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OLLAMA_EMBEDDINGS_AVAILABLE = False

from backend.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_EMBEDDING_MODEL, OLLAMA_EMBEDDING_MODEL
)

class LLMService:
    """Service for managing LLM and embedding models."""
    
    _llm_instance = None
    _embedding_instance = None
    _current_provider = None
    
    @classmethod
    def get_llm(cls, provider: Optional[str] = None, model: Optional[str] = None):
        """Get LLM instance based on provider."""
        provider = provider or LLM_PROVIDER
        
        if provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment variables")
            
            model_name = model or OPENAI_MODEL
            cls._llm_instance = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                openai_api_key=OPENAI_API_KEY
            )
            cls._current_provider = "openai"
        
        elif provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError(
                    "Ollama not available. "
                    "Install with: pip install langchain-community ollama"
                )
            model_name = model or OLLAMA_MODEL
            cls._llm_instance = ChatOllama(
                model=model_name,
                base_url=OLLAMA_BASE_URL
            )
            cls._current_provider = "ollama"
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        return cls._llm_instance
    
    @classmethod
    def get_embeddings(cls, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Get embeddings instance based on provider.
        
        Note: Proxy support is handled via environment variables:
        - HTTP_PROXY=http://your.proxy:port
        - HTTPS_PROXY=http://your.proxy:port
        Do NOT use proxies= parameter (not supported in new OpenAI SDK).
        """
        provider = provider or LLM_PROVIDER
        
        if provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not set in environment variables")
            
            model_name = model or OPENAI_EMBEDDING_MODEL
            cls._embedding_instance = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=OPENAI_API_KEY
            )
            cls._current_provider = "openai"
        
        elif provider == "ollama":
            if not OLLAMA_EMBEDDINGS_AVAILABLE:
                raise ImportError(
                    "Ollama embeddings not available. "
                    "Ensure langchain-community supports Ollama embeddings."
                )
            
            model_name = model or OLLAMA_EMBEDDING_MODEL
            cls._embedding_instance = OllamaEmbeddings(
                model=model_name,
                base_url=OLLAMA_BASE_URL
            )
            cls._current_provider = "ollama"
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
        
        return cls._embedding_instance
    
    @classmethod
    def get_current_provider(cls):
        """Get current LLM provider."""
        return cls._current_provider or LLM_PROVIDER

