"""RAG chain for question answering.

Uses LCEL (LangChain Expression Language) rather than the legacy
`RetrievalQA` chain, which was removed in LangChain 1.0+.
"""
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate

from backend.llm_service import LLMService
from backend.vector_store import VectorStoreManager

class RAGChain:
    """RAG chain for answering questions based on documents."""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

    def _create_qa_prompt(self) -> PromptTemplate:
        """Create prompt template for question answering."""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant that answers questions based on the provided context from documents.
Use only the information from the context to answer the question. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        )
        return prompt

    def answer_question(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Answer a question using RAG.

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        retriever = self.vector_store_manager.get_retriever(doc_ids=doc_ids)
        qa_prompt = self._create_qa_prompt()

        def _build_and_run():
            llm = LLMService.get_llm(model=model)
            source_docs = retriever.invoke(question)

            if doc_ids:
                source_docs = [
                    doc for doc in source_docs
                    if doc.metadata.get("doc_id") in doc_ids
                ]

            context = "\n\n".join(doc.page_content for doc in source_docs)
            chain = qa_prompt | llm
            response = chain.invoke({"context": context, "question": question})
            answer = response.content if hasattr(response, "content") else str(response)
            return answer, source_docs

        answer, source_docs = LLMService.run_with_failover(_build_and_run)

        # Format sources
        sources = []
        seen_sources = set()

        for doc in source_docs:
            doc_id = doc.metadata.get("doc_id")
            filename = doc.metadata.get("filename", "unknown")
            chunk_index = doc.metadata.get("chunk_index", 0)

            source_key = f"{doc_id}_{chunk_index}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                })

        return {
            "answer": answer,
            "sources": sources,
            "query": question
        }
