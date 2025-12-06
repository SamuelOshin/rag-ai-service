from pydantic import BaseModel
from datetime import datetime
from typing import List

class QueryRequest(BaseModel):
    """Request schema for querying documents.

    Attributes:
        question (str): The user's question.
        k (int): Number of chunks to retrieve.
    """
    question: str
    k: int = 3  # Number of chunks to retrieve

class ChunkMetadata(BaseModel):
    """Metadata for a retrieved document chunk.

    Attributes:
        text (str): The text content of the chunk.
        score (float): Similarity score of the chunk.
        doc_id (int): ID of the document the chunk belongs to.
    """
    text: str
    score: float
    doc_id: int

class QueryResponse(BaseModel):
    """Response schema for query results.

    Attributes:
        answer (str): The generated answer.
        sources (List[ChunkMetadata]): List of source chunks.
    """
    answer: str
    sources: List[ChunkMetadata]

class DocumentResponse(BaseModel):
    """Response schema for document information.

    Attributes:
        id (int): Document ID.
        filename (str): Document filename.
        upload_date (datetime): Upload timestamp.
        chunk_count (int): Number of chunks.
    """
    id: int
    filename: str
    upload_date: datetime
    chunk_count: int