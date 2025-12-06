from fastapi import Depends
from app.services.document_service import DocumentProcessor
from app.services.llm_service import OpenRouterService
from app.services.vector_service import VectorDBService
from app.services.rag_service import RAGService

_doc_processor = DocumentProcessor()
_llm_service = OpenRouterService()
_vector_service = VectorDBService()

def get_doc_processor() -> DocumentProcessor:
    """Provides a singleton DocumentProcessor instance.

    Returns:
        DocumentProcessor: The document processor instance.

    Examples:
        >>> proc = get_doc_processor()
        >>> isinstance(proc, DocumentProcessor)
        True
    """
    return _doc_processor

def get_llm_service() -> OpenRouterService:
    """Provides a singleton OpenRouterService instance.

    Returns:
        OpenRouterService: The LLM service instance.

    Examples:
        >>> llm = get_llm_service()
        >>> isinstance(llm, OpenRouterService)
        True
    """
    return _llm_service

def get_vector_service() -> VectorDBService:
    """Provides a singleton VectorDBService instance.

    Returns:
        VectorDBService: The vector database service instance.

    Examples:
        >>> vec = get_vector_service()
        >>> isinstance(vec, VectorDBService)
        True
    """
    return _vector_service

def get_rag_service(
    llm_service: OpenRouterService = Depends(get_llm_service),
    vector_service: VectorDBService = Depends(get_vector_service)
) -> RAGService:
    """Provides the RAG orchestrator service with injected dependencies.

    Args:
        llm_service (OpenRouterService): The LLM service.
        vector_service (VectorDBService): The vector database service.

    Returns:
        RAGService: The RAG service instance.

    Examples:
        >>> rag = get_rag_service()
        >>> isinstance(rag, RAGService)
        True
    """
    return RAGService(llm_service, vector_service)