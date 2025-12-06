from typing import List
from app.services.llm_service import OpenRouterService
from app.services.vector_service import VectorDBService
from app.models.models import QueryResponse, ChunkMetadata

class RAGService:
    """
    Orchestrator service that connects the retrieval (VectorDB) and 
    generation (LLM) layers.
    """
    def __init__(self, llm_service: OpenRouterService, vector_service: VectorDBService):
        self.llm_service = llm_service
        self.vector_service = vector_service

    async def answer_question(self, question: str, k: int = 3) -> QueryResponse:
        # 1. Embed the user's question
        query_embedding = await self.llm_service.get_embedding(question)
        
        # 2. Retrieve relevant chunks from Vector DB
        relevant_chunks = self.vector_service.search(query_embedding, k=k)
        
        if not relevant_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                sources=[]
            )
            
        # 3. Construct the Context for the LLM
        context_text = "\n\n".join([c['text'] for c in relevant_chunks])
        
        # 4. Generate the Answer
        answer = await self.llm_service.generate_answer(context_text, question)
        
        # 5. Format the output
        sources = [
            ChunkMetadata(
                text=c['text'], 
                score=c['score'], 
                doc_id=c['doc_id']
            ) 
            for c in relevant_chunks
        ]
        
        return QueryResponse(answer=answer, sources=sources)