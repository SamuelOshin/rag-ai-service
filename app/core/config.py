from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
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
    return Settings()