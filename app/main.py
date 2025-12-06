from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.database import init_db
from app.core.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create DB tables
    init_db()
    print("Database initialized.")
    yield
    # Shutdown logic if needed

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

app.include_router(router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "RAG Service is running. Visit /docs for Swagger UI."}