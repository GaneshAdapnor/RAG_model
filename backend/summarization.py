"""Document summarization service."""
from typing import List, Optional, Tuple
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from backend.llm_service import LLMService
from backend.config import SUMMARY_MAX_LENGTH
from backend.vector_store import VectorStoreManager

class SummarizationService:
    """Service for generating document summaries and key points."""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
    
    def _create_summary_prompt(self, max_length: int = SUMMARY_MAX_LENGTH) -> PromptTemplate:
        """Create prompt template for summarization."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template=f"""Summarize the following text in approximately {max_length} words. 
Focus on the main ideas and key information.

Text:
{{text}}

Summary:"""
        )
        return prompt
    
    def _create_key_points_prompt(self) -> PromptTemplate:
        """Create prompt template for key points extraction."""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract the key points from the following text. 
Provide them as a bulleted list, with each point being concise and informative.

Text:
{text}

Key Points:"""
        )
        return prompt
    
    def summarize_document(
        self,
        doc_id: str,
        max_length: Optional[int] = None
    ) -> Tuple[str, List[str]]:
        """Summarize a document and extract key points.
        
        Returns:
            Tuple of (summary, key_points)
        """
        # Get document metadata
        doc_metadata = self.vector_store_manager.get_document_metadata(doc_id)
        if not doc_metadata:
            raise ValueError(f"Document {doc_id} not found")
        
        # Retrieve all chunks for this document
        retriever = self.vector_store_manager.get_retriever(doc_ids=[doc_id])
        docs = retriever.get_relevant_documents("summary of the entire document")
        
        # Combine all chunks
        full_text = "\n\n".join([doc.page_content for doc in docs])
        
        if not full_text:
            raise ValueError(f"No content found for document {doc_id}")
        
        # Get LLM
        llm = LLMService.get_llm()
        
        # Generate summary
        max_len = max_length or SUMMARY_MAX_LENGTH
        summary_prompt = self._create_summary_prompt(max_len)
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
        
        summary_result = summary_chain.run(text=full_text[:10000])  # Limit text length
        summary = summary_result.strip()
        
        # Extract key points
        key_points_prompt = self._create_key_points_prompt()
        key_points_chain = LLMChain(llm=llm, prompt=key_points_prompt)
        
        key_points_result = key_points_chain.run(text=full_text[:10000])
        
        # Parse key points (assuming bullet points)
        key_points = []
        for line in key_points_result.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                point = line.lstrip('-•*').strip()
                if point:
                    key_points.append(point)
        
        # If no bullet points found, try splitting by newlines
        if not key_points:
            key_points = [point.strip() for point in key_points_result.split('\n') 
                         if point.strip() and len(point.strip()) > 10]
        
        return summary, key_points[:10]  # Limit to 10 key points

