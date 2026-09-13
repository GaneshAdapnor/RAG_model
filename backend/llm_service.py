"""LLM service supporting OpenRouter (primary), Groq, OpenAI, and Ollama."""
import logging
from typing import Callable, Optional, TypeVar

from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_openai import ChatOpenAI
    OPENAI_COMPATIBLE_AVAILABLE = True
except ImportError:
    OPENAI_COMPATIBLE_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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
    OPENROUTER_API_KEYS as _DEFAULT_OPENROUTER_API_KEYS,
    OPENROUTER_MODEL, OPENROUTER_BASE_URL,
    GROQ_API_KEYS as _DEFAULT_GROQ_API_KEYS, GROQ_MODEL,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)

RATE_LIMIT_MARKERS = ("429", "rate limit", "rate_limit", "too many requests")

# Providers that support multiple API keys with automatic failover on a
# rate-limit error. Each maps to the env-configured default key list.
_MULTI_KEY_PROVIDERS = {
    "openrouter": _DEFAULT_OPENROUTER_API_KEYS,
    "groq": _DEFAULT_GROQ_API_KEYS,
}

T = TypeVar("T")


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)


class NoAPIKeysError(RuntimeError):
    """Raised when a provider is selected but no API key is configured for it."""


class AllAPIKeysExhaustedError(RuntimeError):
    """Raised when every configured API key for a provider has hit a rate limit."""


# Kept as aliases so any existing references to the old Groq-specific names
# keep working.
NoGroqKeysError = NoAPIKeysError
AllGroqKeysExhaustedError = AllAPIKeysExhaustedError


class LLMService:
    """Service for managing LLM and embedding models.

    OpenRouter is the primary/default provider (OpenAI-compatible API,
    access to many models including free/cheap ones like DeepSeek). Groq is
    kept as a secondary option. Both support multiple API keys
    (e.g. OPENROUTER_API_KEYS, comma-separated) with automatic failover to
    the next key on a rate-limit error, via `run_with_failover`.
    """

    _llm_instance = None
    _embedding_instance = None
    _current_provider = None
    _key_indices: dict = {}
    _key_overrides: dict = {}

    @classmethod
    def set_api_keys(cls, provider: str, api_keys: list) -> None:
        """Override a provider's API key list at runtime (e.g. from a UI
        text input) instead of relying solely on the env var captured when
        backend.config was first imported. Resets key rotation to the first key.
        """
        provider = provider.lower()
        cls._key_overrides[provider] = [k.strip() for k in api_keys if k and k.strip()]
        cls._key_indices[provider] = 0

    @classmethod
    def set_groq_api_keys(cls, api_keys: list) -> None:
        """Back-compat alias for set_api_keys('groq', ...)."""
        cls.set_api_keys("groq", api_keys)

    @classmethod
    def _get_api_keys(cls, provider: str) -> list:
        if provider in cls._key_overrides:
            return cls._key_overrides[provider]
        return _MULTI_KEY_PROVIDERS.get(provider, [])

    @classmethod
    def get_llm(cls, provider: Optional[str] = None, model: Optional[str] = None):
        """Get LLM instance based on provider."""
        provider = (provider or LLM_PROVIDER).lower()

        if provider == "openrouter":
            if not OPENAI_COMPATIBLE_AVAILABLE:
                raise ImportError(
                    "OpenRouter support not available. Install with: pip install langchain-openai"
                )
            api_keys = cls._get_api_keys("openrouter")
            if not api_keys:
                raise NoAPIKeysError("OPENROUTER_API_KEY / OPENROUTER_API_KEYS not set in environment variables")

            model_name = model or OPENROUTER_MODEL
            index = cls._key_indices.get("openrouter", 0)
            api_key = api_keys[index % len(api_keys)]
            cls._llm_instance = ChatOpenAI(
                model=model_name,
                temperature=0.0,
                openai_api_key=api_key,
                openai_api_base=OPENROUTER_BASE_URL,
            )
            cls._current_provider = "openrouter"

        elif provider == "groq":
            if not GROQ_AVAILABLE:
                raise ImportError(
                    "Groq support not available. Install with: pip install langchain-groq groq"
                )
            api_keys = cls._get_api_keys("groq")
            if not api_keys:
                raise NoAPIKeysError("GROQ_API_KEY / GROQ_API_KEYS not set in environment variables")

            model_name = model or GROQ_MODEL
            index = cls._key_indices.get("groq", 0)
            api_key = api_keys[index % len(api_keys)]
            cls._llm_instance = ChatGroq(model=model_name, temperature=0.0, api_key=api_key)
            cls._current_provider = "groq"

        elif provider == "openai":
            if not OPENAI_COMPATIBLE_AVAILABLE:
                raise ImportError(
                    "OpenAI support not available. Install with: pip install langchain-openai"
                )
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
    def run_with_failover(cls, build_and_call: Callable[[], T], provider: Optional[str] = None) -> T:
        """Run `build_and_call` (which should itself call `get_llm` internally
        so it always picks up the current key index), rotating to the next
        API key and retrying on a rate-limit error.

        Providers without multi-key support (or with only one key
        configured) just run once and propagate any error as-is.
        """
        provider = (provider or LLM_PROVIDER).lower()
        api_keys = cls._get_api_keys(provider)
        if provider not in _MULTI_KEY_PROVIDERS or len(api_keys) <= 1:
            return build_and_call()

        attempts = len(api_keys)
        last_error: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                return build_and_call()
            except Exception as exc:
                last_error = exc
                if is_rate_limit_error(exc) and attempt < attempts - 1:
                    index = cls._key_indices.get(provider, 0)
                    logger.warning(
                        "%s key #%d hit a rate limit, trying next key...",
                        provider, index % len(api_keys) + 1,
                    )
                    cls._key_indices[provider] = index + 1
                    continue
                raise
        raise AllAPIKeysExhaustedError(
            f"All {attempts} {provider} API key(s) hit a rate limit (last error: {last_error})"
        ) from last_error

    @classmethod
    def get_embeddings(cls, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Get embeddings instance.

        Embeddings are always local SentenceTransformers, regardless of which
        LLM provider is selected: OpenRouter/Groq have no embeddings endpoint
        this app relies on, and local embeddings keep this free and avoid an
        extra external dependency.

        Note: Proxy support is handled via environment variables:
        - HTTP_PROXY=http://your.proxy:port
        - HTTPS_PROXY=http://your.proxy:port
        Do NOT use proxies= parameter (not supported in new OpenAI SDK).
        """
        model_name = model or EMBEDDING_MODEL_NAME
        cls._embedding_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )
        return cls._embedding_instance

    @classmethod
    def get_current_provider(cls):
        """Get current LLM provider."""
        return cls._current_provider or LLM_PROVIDER
