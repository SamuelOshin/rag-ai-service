from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from typing import List

from app.core.database import get_session
from app.models.db_models import Document, DocumentResponse, QueryRequest, QueryResponse, ChunkMetadata
from app.services.document_service import DocumentProcessor
from app.services.llm_service import OpenRouterService
from app.services.vector_service import VectorDBService
from app.utils.response_payloads import success_response

router = APIRouter()

# --- Dependency Injection Helpers ---
def get_doc_processor():
    """Provides a DocumentProcessor instance.

    Returns:
        DocumentProcessor: Instance of the document processor.
    """
    return DocumentProcessor()

def get_llm_service():
    """Provides an OpenRouterService instance.

    Returns:
        OpenRouterService: Instance of the LLM service.
    """
    return OpenRouterService()

def get_vector_service():
    """Provides a VectorDBService instance.

    Returns:
        VectorDBService: Instance of the vector database service.
    """
    return VectorDBService()

async def process_document_background(
    doc_id: int,
    text: str,
    db: Session,
    doc_processor: DocumentProcessor,
    llm_service: OpenRouterService,
    vector_service: VectorDBService
):
    """Processes a document in the background by chunking, embedding, and storing data.

    Chunks the text, updates the database with chunk count, generates embeddings,
    and stores chunks in the vector database.

    Args:
        doc_id (int): The document ID.
        text (str): The extracted text from the document.
        db (Session): Database session.
        doc_processor (DocumentProcessor): Document processing service.
        llm_service (OpenRouterService): LLM service for embeddings.
        vector_service (VectorDBService): Vector database service.

    Raises:
        Exception: If any step in processing fails.

    Examples:
        >>> await process_document_background(1, "sample text", session, proc, llm, vec)
        # Processes document 1 in background
    """
    try:
        chunks = doc_processor.chunk_text(text)
        
        doc = db.get(Document, doc_id)
        if doc:
            doc.chunk_count = len(chunks)
            db.add(doc)
            db.commit()
        
        embeddings = []
        for chunk in chunks:
            emb = await llm_service.get_embedding(chunk)
            embeddings.append(emb)
            
        vector_service.upsert_chunks(doc_id, chunks, embeddings)
        print(f"Successfully processed document {doc_id}")
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")

@router.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    doc_processor: DocumentProcessor = Depends(get_doc_processor),
    llm_service: OpenRouterService = Depends(get_llm_service),
    vector_service: VectorDBService = Depends(get_vector_service)
):
    """Uploads a document, extracts text, and initiates background processing.

    Extracts text from the uploaded file, creates a database record,
    and schedules background processing for chunking and embedding.

    Args:
        background_tasks (BackgroundTasks): FastAPI background tasks.
        file (UploadFile): The uploaded file.
        db (Session): Database session.
        doc_processor (DocumentProcessor): Document processing service.
        llm_service (OpenRouterService): LLM service.
        vector_service (VectorDBService): Vector database service.

    Returns:
        Dict: Standardized success response with document data.

    Raises:
        HTTPException: If file processing fails.

    Examples:
        >>> # API call: POST /api/v1/documents/upload with file
        # Returns success response with document details
    """
    text = await doc_processor.process_file(file)
    
    new_doc = Document(
        filename=file.filename,
        content_type=file.content_type,
        file_size=file.size or 0
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    background_tasks.add_task(
        process_document_background, 
        new_doc.id, 
        text, 
        db,
        doc_processor, 
        llm_service, 
        vector_service
    )
    
    return success_response(201, "Document uploaded successfully", data=new_doc.dict())

@router.post("/query")
async def query_documents(
    request: QueryRequest,
    llm_service: OpenRouterService = Depends(get_llm_service),
    vector_service: VectorDBService = Depends(get_vector_service)
):
    """Queries documents using RAG to generate an answer.

    Embeds the question, retrieves relevant chunks, and generates an answer using LLM.

    Args:
        request (QueryRequest): Query request containing question and k.
        llm_service (OpenRouterService): LLM service.
        vector_service (VectorDBService): Vector database service.

    Returns:
        Dict: Standardized success response with query result.

    Examples:
        >>> # API call: POST /api/v1/query with {"question": "What is AI?", "k": 3}
        # Returns success response with answer and sources
    """
    query_embedding = await llm_service.get_embedding(request.question)
    
    relevant_chunks = vector_service.search(query_embedding, k=request.k)
    
    if not relevant_chunks:
        response = QueryResponse(
            answer="No relevant information found in the uploaded documents.",
            sources=[]
        )
    else:
        context_text = "\n\n".join([c['text'] for c in relevant_chunks])
        
        answer = await llm_service.generate_answer(context_text, request.question)
        
        sources = [
            ChunkMetadata(text=c['text'], score=c['score'], doc_id=c['doc_id']) 
            for c in relevant_chunks
        ]
        
        response = QueryResponse(answer=answer, sources=sources)
    
    return success_response(200, "Query processed successfully", data=response.dict())

@router.get("/documents")
def list_documents(db: Session = Depends(get_session)):
    """Lists all uploaded documents.

    Args:
        db (Session): Database session.

    Returns:
        Dict: Standardized success response with list of documents.

    Examples:
        >>> # API call: GET /api/v1/documents
        # Returns success response with document list
    """
    docs = db.exec(select(Document)).all()
    return success_response(200, "Documents retrieved successfully", data=[doc.dict() for doc in docs])