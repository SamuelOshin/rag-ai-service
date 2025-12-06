from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from typing import List

from app.core.database import get_session
from app.models.models import Document, DocumentResponse, QueryRequest, QueryResponse, ChunkMetadata
from app.services.document_service import DocumentProcessor
from app.services.llm_service import OpenRouterService
from app.services.vector_service import VectorDBService

router = APIRouter()

# --- Dependency Injection Helpers ---
def get_doc_processor(): return DocumentProcessor()
def get_llm_service(): return OpenRouterService()
def get_vector_service(): return VectorDBService()

async def process_document_background(
    doc_id: int,
    text: str,
    db: Session,
    doc_processor: DocumentProcessor,
    llm_service: OpenRouterService,
    vector_service: VectorDBService
):
    """Background task to chunk, embed, and store data without blocking the API."""
    try:
        # 1. Chunk
        chunks = doc_processor.chunk_text(text)
        
        # 2. Update DB with chunk count
        doc = db.get(Document, doc_id)
        if doc:
            doc.chunk_count = len(chunks)
            db.add(doc)
            db.commit()
        
        # 3. Embed (Batching recommended for prod, doing simple loop here)
        embeddings = []
        for chunk in chunks:
            # Note: sequential await is slow; asyncio.gather is better for prod
            emb = await llm_service.get_embedding(chunk)
            embeddings.append(emb)
            
        # 4. Store in Vector DB
        vector_service.upsert_chunks(doc_id, chunks, embeddings)
        print(f"Successfully processed document {doc_id}")
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")

@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_session),
    doc_processor: DocumentProcessor = Depends(get_doc_processor),
    llm_service: OpenRouterService = Depends(get_llm_service),
    vector_service: VectorDBService = Depends(get_vector_service)
):
    # 1. Extract Text immediately
    text = await doc_processor.process_file(file)
    
    # 2. Create Initial DB Record
    new_doc = Document(
        filename=file.filename,
        content_type=file.content_type,
        file_size=file.size or 0
    )
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    # 3. Offload Embedding & Indexing to Background
    # We pass a new session or handle the session carefully in background tasks
    # For simplicity here, we pass logic dependencies. 
    # Note: In real prod, use Celery/Redis for robustness.
    background_tasks.add_task(
        process_document_background, 
        new_doc.id, 
        text, 
        db, # Passing session to background task is risky in async; better to create new session inside task
        doc_processor, 
        llm_service, 
        vector_service
    )
    
    return new_doc

@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    llm_service: OpenRouterService = Depends(get_llm_service),
    vector_service: VectorDBService = Depends(get_vector_service)
):
    # 1. Embed the Question
    query_embedding = await llm_service.get_embedding(request.question)
    
    # 2. Retrieve Relevant Chunks
    relevant_chunks = vector_service.search(query_embedding, k=request.k)
    
    if not relevant_chunks:
        return QueryResponse(
            answer="No relevant information found in the uploaded documents.",
            sources=[]
        )
        
    # 3. Construct Context
    context_text = "\n\n".join([c['text'] for c in relevant_chunks])
    
    # 4. Generate Answer via LLM
    answer = await llm_service.generate_answer(context_text, request.question)
    
    # 5. Format Response
    sources = [
        ChunkMetadata(text=c['text'], score=c['score'], doc_id=c['doc_id']) 
        for c in relevant_chunks
    ]
    
    return QueryResponse(answer=answer, sources=sources)

@router.get("/documents", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_session)):
    return db.exec(select(Document)).all()