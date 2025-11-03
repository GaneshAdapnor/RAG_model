"""Vector store management with FAISS persistence."""
import os
import json
import uuid
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from backend.llm_service import LLMService

class VectorStoreManager:
    """Manage FAISS vector store with persistence."""
    
    def __init__(self):
        self.vector_store_path = Path(VECTOR_STORE_PATH)
        self.metadata_path = self.vector_store_path.parent / "metadata.json"
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
        """Remove a document from the vector store."""
        if doc_id not in self.documents_metadata:
            return False
        
        # Note: FAISS doesn't support removing individual documents easily
        # For now, we'll mark it in metadata and rebuild on next use
        # In production, you might want to rebuild the index
        
        del self.documents_metadata[doc_id]
        self._save_metadata()
        
        # TODO: Implement proper document removal
        # For now, we'll just remove from metadata
        # A full rebuild would be needed for complete removal
        
        return True
    
    def get_retriever(self, doc_ids: Optional[List[str]] = None):
        """Get a retriever from the vector store.
        
        Note: FAISS doesn't support direct metadata filtering.
        Filtering by doc_ids is handled post-retrieval in the RAG chain.
        """
        if self.vector_store is None:
            if not self._load_vector_store():
                raise ValueError("No vector store available. Please upload documents first.")
        
        # Store doc_ids filter for post-retrieval filtering
        # FAISS doesn't support metadata filtering directly
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": TOP_K * 2}  # Get more to filter down
        )
        
        # Store filter for use in RAG chain
        retriever.doc_ids_filter = doc_ids
        
        return retriever
    
    def get_documents_metadata(self) -> Dict[str, Dict]:
        """Get all documents metadata."""
        return self.documents_metadata
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a specific document."""
        return self.documents_metadata.get(doc_id)
    
    def initialize(self):
        """Initialize vector store (load if exists)."""
        self._load_vector_store()

