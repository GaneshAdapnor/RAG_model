"""Startup script for the RAG backend API."""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    import uvicorn
    from backend.config import API_HOST, API_PORT
    
    print(f"Starting RAG System API on {API_HOST}:{API_PORT}")
    print(f"API docs available at http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

