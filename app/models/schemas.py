from pydantic import BaseModel
from datetime import datetime
from typing import List
# Request for Querying
class QueryRequest(BaseModel):
    question: str
    k: int = 3 # Number of chunks to retrieve

# Response for a single Chunk
class ChunkMetadata(BaseModel):
    text: str
    score: float
    doc_id: int

# Full Response
class QueryResponse(BaseModel):
    answer: str
    sources: List[ChunkMetadata]

# Document Response
class DocumentResponse(BaseModel):
    id: int
    filename: str
    upload_date: datetime
    chunk_count: int