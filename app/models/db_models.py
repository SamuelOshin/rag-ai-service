from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field


# --- Database Models (SQLModel) ---
class Document(SQLModel, table=True):
    """Represents a document stored in the database.

    Attributes:
        id (Optional[int]): Unique identifier for the document.
        filename (str): Name of the uploaded file.
        content_type (str): MIME type of the file.
        file_size (int): Size of the file in bytes.
        chunk_count (int): Number of chunks created from the document.
        upload_date (datetime): Timestamp of upload.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    content_type: str
    file_size: int
    chunk_count: int = 0
    upload_date: datetime = Field(default_factory=datetime.utcnow)

