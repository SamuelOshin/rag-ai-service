from typing import List
from app.services.llm_service import OpenRouterService
from app.services.vector_service import VectorDBService
from app.models.db_models import QueryResponse, ChunkMetadata

class RAGService:
    """Orchestrator service connecting retrieval and generation layers."""

    def __init__(self, llm_service: OpenRouterService, vector_service: VectorDBService):
        """Initializes the RAG service with required dependencies.

        Args:
            llm_service (OpenRouterService): Service for LLM interactions.
            vector_service (VectorDBService): Service for vector database operations.
        """
        self.llm_service = llm_service
        self.vector_service = vector_service

    async def answer_question(self, question: str, k: int = 3) -> QueryResponse:
        """Answers a question using RAG by retrieving and generating.

        Args:
            question (str): The user's question.
            k (int): Number of chunks to retrieve.

        Returns:
            QueryResponse: Response containing answer and sources.

        Examples:
            >>> response = await rag.answer_question("What is machine learning?")
            >>> response.answer
            'Machine learning is...'
        """
        query_embedding = await self.llm_service.get_embedding(question)
        
        relevant_chunks = self.vector_service.search(query_embedding, k=k)
        
        if not relevant_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                sources=[]
            )
            
        context_text = "\n\n".join([c['text'] for c in relevant_chunks])
        
        answer = await self.llm_service.generate_answer(context_text, question)
        
        sources = [
            ChunkMetadata(
                text=c['text'], 
                score=c['score'], 
                doc_id=c['doc_id']
            ) 
            for c in relevant_chunks
        ]
        
        return QueryResponse(answer=answer, sources=sources)