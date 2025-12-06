from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.database import init_db
from app.core.config import get_settings

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the application lifespan events.

    Initializes the database on startup and handles shutdown logic.

    Yields:
        None
    """
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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handles Pydantic validation errors and returns standardized error response.

    Args:
        request (Request): The incoming request.
        exc (RequestValidationError): The validation exception.

    Returns:
        JSONResponse: Standardized error response.
    """
    errors = {}
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error['loc'])
        msg = error['msg']
        if field not in errors:
            errors[field] = []
        errors[field].append(msg)
    return JSONResponse(
        status_code=400,
        content={
            "status": "failure",
            "status_code": 400,
            "message": "Validation failed",
            "error_code": "VALIDATION_ERROR",
            "errors": errors
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles HTTP exceptions and returns standardized error response.

    Args:
        request (Request): The incoming request.
        exc (HTTPException): The HTTP exception.

    Returns:
        JSONResponse: Standardized error response.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "failure",
            "status_code": exc.status_code,
            "message": exc.detail,
            "errors": {}
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handles general exceptions and returns standardized error response.

    Args:
        request (Request): The incoming request.
        exc (Exception): The general exception.

    Returns:
        JSONResponse: Standardized error response.
    """
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "status_code": 500,
            "message": "Internal server error",
            "errors": {}
        }
    )

@app.get("/")
def root():
    """Root endpoint that provides a welcome message.

    Returns:
        Dict[str, str]: Welcome message dictionary.

    Examples:
        >>> root()
        {'message': 'RAG Service is running. Visit /docs for Swagger UI.'}
    """
    return {"message": "RAG Service is running. Visit /docs for Swagger UI."}