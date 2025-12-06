from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field


# --- Database Models (SQLModel) ---
class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    content_type: str
    file_size: int
    chunk_count: int = 0
    upload_date: datetime = Field(default_factory=datetime.utcnow)

