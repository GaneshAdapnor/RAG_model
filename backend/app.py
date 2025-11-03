"""FastAPI application for RAG system."""
import os
import uuid
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import (
    UPLOAD_FOLDER, CORS_ORIGINS, AUTO_SUMMARIZE, LLM_PROVIDER
)
from backend.models import (
    DocumentUploadResponse, QueryRequest, QueryResponse,
    SummarizeRequest, SummarizeResponse, DocumentInfo,
    DocumentsListResponse, ConfigUpdateRequest, ConfigResponse
)
from backend.document_processor import DocumentProcessor
from backend.vector_store import VectorStoreManager
from backend.llm_service import LLMService
from backend.summarization import SummarizationService
from backend.rag_chain import RAGChain

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API for document Q&A",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_store_manager = VectorStoreManager()
document_processor = DocumentProcessor()
summarization_service = SummarizationService(vector_store_manager)
rag_chain = RAGChain(vector_store_manager)

# Initialize upload folder
upload_path = Path(UPLOAD_FOLDER)
upload_path.mkdir(parents=True, exist_ok=True)

# Initialize vector store on startup
vector_store_manager.initialize()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/api/upload",
            "query": "/api/query",
            "summarize": "/api/summarize",
            "documents": "/api/documents",
            "config": "/api/config"
        }
    }


@app.post("/api/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    auto_summarize: bool = Form(default=None)
):
    """Upload and process a document."""
    try:
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_extension = Path(file.filename).suffix
        saved_filename = f"{doc_id}{file_extension}"
        file_path = upload_path / saved_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        try:
            text, file_type = document_processor.process_document(
                str(file_path),
                file.filename
            )
        except Exception as e:
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=400, detail=str(e))
        
        # Add to vector store
        try:
            chunks_created = vector_store_manager.add_document(
                doc_id=doc_id,
                text=text,
                filename=file.filename,
                file_type=file_type
            )
        except Exception as e:
            # Clean up file on error
            if file_path.exists():
                file_path.unlink()
            raise HTTPException(status_code=500, detail=f"Error adding to vector store: {str(e)}")
        
        # Auto-summarize if enabled
        summary = None
        key_points = None
        should_summarize = auto_summarize if auto_summarize is not None else AUTO_SUMMARIZE
        
        if should_summarize:
            try:
                summary, key_points = summarization_service.summarize_document(doc_id)
                
                # Save summary to metadata
                doc_metadata = vector_store_manager.get_document_metadata(doc_id)
                if doc_metadata:
                    doc_metadata["summary"] = summary
                    doc_metadata["key_points"] = key_points
                    vector_store_manager._save_metadata()
            except Exception as e:
                # Don't fail upload if summarization fails
                print(f"Warning: Summarization failed: {e}")
        
        return DocumentUploadResponse(
            doc_id=doc_id,
            filename=file.filename,
            file_type=file_type,
            status="success",
            text_length=len(text),
            chunks_created=chunks_created,
            summary=summary,
            key_points=key_points,
            message=f"Document uploaded and processed successfully. Created {chunks_created} chunks."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question about the documents."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if vector store has documents
        metadata = vector_store_manager.get_documents_metadata()
        if not metadata:
            raise HTTPException(
                status_code=400,
                detail="No documents uploaded. Please upload documents first."
            )
        
        result = rag_chain.answer_question(
            question=request.question,
            doc_ids=request.doc_ids
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=result["query"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize_document(request: SummarizeRequest):
    """Generate or regenerate summary for a document."""
    try:
        summary, key_points = summarization_service.summarize_document(
            doc_id=request.doc_id,
            max_length=request.max_length
        )
        
        # Save to metadata
        doc_metadata = vector_store_manager.get_document_metadata(request.doc_id)
        if doc_metadata:
            doc_metadata["summary"] = summary
            doc_metadata["key_points"] = key_points
            vector_store_manager._save_metadata()
        
        return SummarizeResponse(
            doc_id=request.doc_id,
            summary=summary,
            key_points=key_points
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.get("/api/documents", response_model=DocumentsListResponse)
async def list_documents():
    """List all uploaded documents."""
    try:
        metadata = vector_store_manager.get_documents_metadata()
        
        documents = []
        for doc_id, doc_data in metadata.items():
            documents.append(DocumentInfo(
                doc_id=doc_id,
                filename=doc_data.get("filename", "unknown"),
                file_type=doc_data.get("file_type", "unknown"),
                upload_date=doc_data.get("upload_date", ""),
                text_length=doc_data.get("text_length", 0),
                chunks=doc_data.get("chunks", 0),
                has_summary="summary" in doc_data
            ))
        
        return DocumentsListResponse(
            documents=documents,
            total=len(documents)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document."""
    try:
        # Remove from vector store
        success = vector_store_manager.remove_document(doc_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Delete uploaded file
        for file_path in upload_path.glob(f"{doc_id}.*"):
            file_path.unlink()
        
        return {"message": f"Document {doc_id} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current LLM configuration."""
    try:
        provider = LLMService.get_current_provider()
        
        from backend.config import (
            OPENAI_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL
        )
        
        return ConfigResponse(
            llm_provider=provider,
            openai_model=OPENAI_MODEL if provider == "openai" else None,
            ollama_model=OLLAMA_MODEL if provider == "ollama" else None,
            ollama_base_url=OLLAMA_BASE_URL if provider == "ollama" else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")


@app.post("/api/config", response_model=ConfigResponse)
async def update_config(request: ConfigUpdateRequest):
    """Update LLM configuration."""
    try:
        # Update configuration
        if request.llm_provider:
            os.environ["LLM_PROVIDER"] = request.llm_provider
        
        if request.openai_model:
            os.environ["OPENAI_MODEL"] = request.openai_model
        
        if request.ollama_model:
            os.environ["OLLAMA_MODEL"] = request.ollama_model
        
        if request.ollama_base_url:
            os.environ["OLLAMA_BASE_URL"] = request.ollama_base_url
        
        # Reload config
        import importlib
        import backend.config
        importlib.reload(backend.config)
        
        # Get updated config
        provider = LLMService.get_current_provider()
        
        from backend.config import (
            OPENAI_MODEL, OLLAMA_MODEL, OLLAMA_BASE_URL
        )
        
        return ConfigResponse(
            llm_provider=provider,
            openai_model=OPENAI_MODEL if provider == "openai" else None,
            ollama_model=OLLAMA_MODEL if provider == "ollama" else None,
            ollama_base_url=OLLAMA_BASE_URL if provider == "ollama" else None
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating config: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    from backend.config import API_HOST, API_PORT
    
    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

