from fastapi import Depends
from app.services.document_service import DocumentProcessor
from app.services.llm_service import OpenRouterService
from app.services.vector_service import VectorDBService
from app.services.rag_service import RAGService

# Singleton instances (optional, but good for caching clients)
_doc_processor = DocumentProcessor()
_llm_service = OpenRouterService()
_vector_service = VectorDBService()

def get_doc_processor() -> DocumentProcessor:
    return _doc_processor

def get_llm_service() -> OpenRouterService:
    return _llm_service

def get_vector_service() -> VectorDBService:
    return _vector_service

def get_rag_service(
    llm_service: OpenRouterService = Depends(get_llm_service),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> RAGService:
    """
    Dependency that provides the RAG orchestrator, 
    automatically injecting the required sub-services.
    """
    return RAGService(llm_service, vector_service)