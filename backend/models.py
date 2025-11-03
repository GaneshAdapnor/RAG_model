"""Pydantic models for API requests and responses."""
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    """Response after document upload."""
    doc_id: str
    filename: str
    file_type: str
    status: str
    text_length: int
    chunks_created: int
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    message: str

class QueryRequest(BaseModel):
    """Request for question answering."""
    question: str
    doc_ids: Optional[List[str]] = None  # Filter to specific documents

class QueryResponse(BaseModel):
    """Response from question answering."""
    answer: str
    sources: List[Dict[str, Any]]
    query: str

class SummarizeRequest(BaseModel):
    """Request for document summarization."""
    doc_id: str
    max_length: Optional[int] = None

class SummarizeResponse(BaseModel):
    """Response from summarization."""
    doc_id: str
    summary: str
    key_points: List[str]

class DocumentInfo(BaseModel):
    """Document information."""
    doc_id: str
    filename: str
    file_type: str
    upload_date: str
    text_length: int
    chunks: int
    has_summary: bool

class DocumentsListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentInfo]
    total: int

class ConfigUpdateRequest(BaseModel):
    """Request to update LLM configuration."""
    llm_provider: Optional[Literal["openai", "ollama"]] = None
    openai_model: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_base_url: Optional[str] = None

class ConfigResponse(BaseModel):
    """Current configuration."""
    llm_provider: str
    openai_model: Optional[str] = None
    ollama_model: Optional[str] = None
    ollama_base_url: Optional[str] = None

