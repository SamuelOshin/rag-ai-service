from sqlmodel import SQLModel, Session, create_engine
from app.core.config import get_settings

settings = get_settings()

# We use echo=True for debugging locally, set to False in production
engine = create_engine(settings.DATABASE_URL, echo=False)

def init_db():
    """Creates all database tables based on SQLModel definitions.

    Examples:
        >>> init_db()
        # Creates tables in the database
    """
    SQLModel.metadata.create_all(engine)

def get_session():
    """Provides a database session as a context manager.

    Yields:
        Session: A SQLModel session for database operations.

    Examples:
        >>> with get_session() as session:
        ...     # Perform database operations
        ...     pass
    """
    with Session(engine) as session:
        yield session