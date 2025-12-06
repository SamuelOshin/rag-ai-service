from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        PROJECT_NAME (str): Name of the project.
        API_V1_STR (str): API version string.
        DATABASE_URL (str): Database connection URL.
        CHROMA_HOST (str): Chroma database host.
        CHROMA_PORT (int): Chroma database port.
        COLLECTION_NAME (str): Chroma collection name.
        OPENROUTER_API_KEY (str): API key for OpenRouter.
        OPENROUTER_BASE_URL (str): Base URL for OpenRouter.
        EMBEDDING_MODEL (str): Model for embeddings.
        LLM_MODEL (str): Model for LLM.
    """
    # App Settings
    PROJECT_NAME: str = "RAG Document Service"
    API_V1_STR: str = "/api/v1"
    
    # Database (Postgres)
    DATABASE_URL: str = "postgresql://user:password@db:5432/rag_db"
    
    # Vector DB (Chroma)
    CHROMA_HOST: str = "chroma"
    CHROMA_PORT: int = 8000
    COLLECTION_NAME: str = "rag_documents"
    
    # OpenRouter / LLM
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Model Selection (Configurable)
    EMBEDDING_MODEL: str = "openai/text-embedding-3-small"
    LLM_MODEL: str = "meta-llama/llama-3.1-70b-instruct"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    """Retrieves cached application settings.

    Returns:
        Settings: The application settings instance.

    Examples:
        >>> settings = get_settings()
        >>> settings.PROJECT_NAME
        'RAG Document Service'
    """
    return Settings()