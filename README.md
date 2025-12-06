# RAG AI Service

A Retrieval-Augmented Generation (RAG) AI service built with FastAPI, designed to process documents and answer questions based on their content using advanced language models and vector search.

## Features

- **Document Processing**: Supports PDF and DOCX file uploads with automatic text extraction
- **Intelligent Chunking**: Splits documents into semantically meaningful chunks using LangChain
- **Vector Storage**: Uses ChromaDB for efficient vector storage and similarity search
- **LLM Integration**: Leverages OpenRouter API for embeddings and answer generation
- **RESTful API**: Clean, documented endpoints for document management and querying
- **Background Processing**: Asynchronous document processing to handle large files
- **Metadata Storage**: PostgreSQL database for document metadata and tracking
- **Docker Support**: Containerized deployment with Docker Compose

## File Structure

```
rag_ai_service/
├── docker-compose.yml          # Docker Compose configuration for multi-service setup
├── Dockerfile                  # Docker build instructions for the FastAPI app
├── pyproject.toml              # Project configuration and dependencies (uv)
├── requirements.txt            # Python dependencies for Docker build
├── README.md                   # This file
└── app/
    ├── __init__.py
    ├── main.py                 # FastAPI application entry point with lifespan management
    ├── api/
    │   ├── __init__.py
    │   ├── dependencies.py     # Dependency injection helpers
    │   ├── routes.py           # API route definitions
    │   └── __pycache__/
    ├── core/
    │   ├── __init__.py
    │   ├── config.py           # Application settings and environment variables
    │   └── database.py         # Database initialization and session management
    ├── models/
    │   ├── __init__.py
    │   ├── db_models.py        # SQLModel database schemas
    │   └── schemas.py          # Pydantic request/response schemas
    ├── services/
    │   ├── __init__.py
    │   ├── document_service.py # Document processing and text extraction
    │   ├── llm_service.py      # OpenRouter API integration for LLM and embeddings
    │   ├── rag_service.py      # RAG pipeline orchestration
    │   └── vector_service.py   # ChromaDB vector database operations
    └── utils/
        ├── __init__.py
        └── response_payloads.py # Standardized API response utilities
```

## Prerequisites

- Python 3.10 or higher
- uv package manager (recommended) or pip
- Docker and Docker Compose (for containerized deployment)
- OpenRouter API key (for LLM access)

## Installation and Setup

### Option 1: Using uv (Recommended)

1. **Install uv**:
   ```bash
   # Windows
   winget install astral-sh.uv

   # macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/SamuelOshin/rag-ai-service.git
   cd rag-ai-service
   ```

3. **Create virtual environment and install dependencies**:
   ```bash
   uv venv
   uv sync
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
   CHROMA_HOST=localhost
   CHROMA_PORT=8000
   COLLECTION_NAME=rag_documents
   ```

5. **Start ChromaDB** (for local vector storage):
   ```bash
   docker run -p 8000:8000 chromadb/chroma:latest
   ```

6. **Start PostgreSQL** (for metadata storage):
   ```bash
   docker run -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_db -p 5432:5432 postgres:15-alpine
   ```

### Option 2: Using Docker Compose (Full Stack)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SamuelOshin/rag-ai-service.git
   cd rag-ai-service
   ```

2. **Create environment file**:
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

3. **Start all services**:
   ```bash
   docker-compose up --build
   ```

   This will start:
   - FastAPI backend on `http://localhost:8000`
   - PostgreSQL database on `localhost:5432`
   - ChromaDB on `localhost:8001`

## Usage

### Starting the Application

#### With uv:
```bash
uv run uvicorn app.main:app --reload
```

#### With Docker Compose:
```bash
docker-compose up
```

The API will be available at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

### API Endpoints

#### Health Check
- **GET** `/`
- Returns a welcome message and API status

#### Upload Document
- **POST** `/api/v1/documents/upload`
- Uploads a PDF or DOCX file for processing
- **Request**: Multipart form data with `file` field
- **Response**: Document metadata and processing status

#### Query Documents
- **POST** `/api/v1/query`
- Queries the RAG system with a question
- **Request Body**:
  ```json
  {
    "question": "What is the main topic of the document?",
    "k": 3
  }
  ```
- **Response**: Generated answer with source chunks and relevance scores

#### List Documents
- **GET** `/api/v1/documents`
- Returns a list of all uploaded documents with metadata

### Example Usage

1. **Upload a document**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/documents/upload" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@sample_document.pdf"
   ```

2. **Query the system**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/query" \
        -H "Content-Type: application/json" \
        -d '{
          "question": "What are the key findings?",
          "k": 5
        }'
   ```

3. **List documents**:
   ```bash
   curl -X GET "http://localhost:8000/api/v1/documents" \
        -H "accept: application/json"
   ```

## Configuration

The application uses the following environment variables (defined in `app/core/config.py`):

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | API key for OpenRouter | Required |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@db:5432/rag_db` |
| `CHROMA_HOST` | ChromaDB host | `chroma` (Docker) / `localhost` (local) |
| `CHROMA_PORT` | ChromaDB port | `8000` |
| `COLLECTION_NAME` | ChromaDB collection name | `rag_documents` |
| `EMBEDDING_MODEL` | Model for text embeddings | `openai/text-embedding-3-small` |
| `LLM_MODEL` | Model for answer generation | `meta-llama/llama-3.1-70b-instruct` |

## Development

### Running Tests
```bash
uv run pytest
```

### Code Formatting
```bash
uv run black .
uv run isort .
```

### Linting
```bash
uv run flake8 .
```

## Architecture Overview

The RAG AI Service follows a modular architecture:

1. **Document Ingestion**: Files are uploaded via the API and processed asynchronously
2. **Text Extraction**: PDF/DOCX files are parsed to extract raw text
3. **Chunking**: Text is split into semantically coherent chunks
4. **Embedding**: Chunks are converted to vector embeddings using OpenRouter
5. **Storage**: Embeddings are stored in ChromaDB, metadata in PostgreSQL
6. **Query Processing**: User questions are embedded and used for vector similarity search
7. **Answer Generation**: Relevant chunks are fed to the LLM to generate contextual answers

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.