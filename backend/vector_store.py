"""Vector store management with FAISS persistence."""
import os
import json
import shutil
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# Try newer import first, fallback to older
try:
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain.schema import Document
    except ImportError:
        raise ImportError("Could not import Document. Please install langchain-core or langchain.")
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try to import from backend.config, fallback to defaults if not available
try:
    from backend.config import STORAGE_ROOT, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
except ImportError:
    # Default values if backend.config is not available
    STORAGE_ROOT = Path("./backend/storage")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K = 5

# Try to import LLMService, but it's only used in backend context
try:
    from backend.llm_service import LLMService
    LLM_SERVICE_AVAILABLE = True
except ImportError:
    LLM_SERVICE_AVAILABLE = False
    # Create a dummy class to prevent errors
    class LLMService:
        @staticmethod
        def get_embeddings():
            raise NotImplementedError("LLMService not available in this context")

class VectorStoreManager:
    """Manage FAISS vector store with persistence.

    Every instance is scoped to a `session_id` so that documents uploaded by
    one visitor are never visible to another. Each session gets its own
    directory under STORAGE_ROOT/sessions/<session_id>/, containing its own
    FAISS index and metadata.json. Omitting session_id falls back to a
    single shared "default" session, which is only appropriate for local,
    single-user use (e.g. the standalone FastAPI backend run without a
    session-aware client).
    """

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        session_dir = Path(STORAGE_ROOT) / "sessions" / session_id
        self.vector_store_path = session_dir / "faiss_index"
        self.metadata_path = session_dir / "metadata.json"
        self.documents_metadata: Dict[str, Dict] = {}
        self.vector_store: Optional[FAISS] = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load documents metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.documents_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                self.documents_metadata = {}
    
    def _save_metadata(self):
        """Save documents metadata to disk."""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.documents_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def _load_vector_store(self) -> bool:
        """Load existing vector store from disk."""
        if self.vector_store_path.exists():
            try:
                if not LLM_SERVICE_AVAILABLE:
                    return False
                embeddings = LLMService.get_embeddings()
                self.vector_store = FAISS.load_local(
                    str(self.vector_store_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception as e:
                print(f"Warning: Could not load vector store: {e}")
                return False
        return False
    
    def _save_vector_store(self):
        """Save vector store to disk."""
        if self.vector_store:
            try:
                self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(self.vector_store_path))
            except Exception as e:
                print(f"Warning: Could not save vector store: {e}")
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        filename: str,
        file_type: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """Add a document to the vector store.
        
        Returns:
            Number of chunks created
        """
        # Split document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        chunk_count = len(chunks)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "doc_id": doc_id,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": chunk_count
            }
            if metadata:
                chunk_metadata.update(metadata)
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))
        
        # Get embeddings
        if not LLM_SERVICE_AVAILABLE:
            raise NotImplementedError("LLMService not available. This module is for backend use only.")
        embeddings = LLMService.get_embeddings()
        
        # Add to vector store
        if self.vector_store is None:
            # Create new vector store
            if not self._load_vector_store():
                self.vector_store = FAISS.from_documents(documents, embeddings)
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)
        
        # Save metadata
        self.documents_metadata[doc_id] = {
            "filename": filename,
            "file_type": file_type,
            "upload_date": datetime.now().isoformat(),
            "text_length": len(text),
            "chunks": chunk_count
        }
        self._save_metadata()
        
        # Save vector store
        self._save_vector_store()
        
        return chunk_count
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and rebuild the FAISS index without its chunks.

        FAISS itself doesn't support deleting by metadata filter, so this
        rebuilds the index from the remaining documents in the docstore.
        Cheap enough for the per-session, moderate-document-count use case
        this app targets.
        """
        if doc_id not in self.documents_metadata:
            return False

        del self.documents_metadata[doc_id]
        self._save_metadata()

        if self.vector_store is not None:
            remaining_docs = [
                doc for doc in self.vector_store.docstore._dict.values()
                if doc.metadata.get("doc_id") != doc_id
            ]
            if remaining_docs:
                embeddings = LLMService.get_embeddings()
                self.vector_store = FAISS.from_documents(remaining_docs, embeddings)
                self._save_vector_store()
            else:
                self.vector_store = None
                self._delete_vector_store_files()

        return True

    def clear_all(self):
        """Remove every document, its vectors, and its metadata for this session."""
        self.documents_metadata = {}
        self._save_metadata()
        self.vector_store = None
        self._delete_vector_store_files()

    def _delete_vector_store_files(self):
        if self.vector_store_path.exists():
            shutil.rmtree(self.vector_store_path, ignore_errors=True)
    
    def get_retriever(self, doc_ids: Optional[List[str]] = None):
        """Get a retriever from the vector store.
        
        Note: FAISS doesn't support direct metadata filtering.
        Filtering by doc_ids is handled post-retrieval in the RAG chain.
        """
        if self.vector_store is None:
            if not self._load_vector_store():
                raise ValueError("No vector store available. Please upload documents first.")

        # FAISS doesn't support metadata filtering directly, so we retrieve
        # extra candidates and let the caller (e.g. RAGChain) filter by
        # doc_ids from each Document's metadata after retrieval.
        k = TOP_K * 2 if doc_ids else TOP_K
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def get_documents_metadata(self) -> Dict[str, Dict]:
        """Get all documents metadata."""
        return self.documents_metadata
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a specific document."""
        return self.documents_metadata.get(doc_id)
    
    def initialize(self):
        """Initialize vector store (load if exists)."""
        self._load_vector_store()

