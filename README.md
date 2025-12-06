# RAG AI Service

A Retrieval-Augmented Generation (RAG) AI service built with FastAPI.

## Features

- Document parsing and chunking
- LLM integration with OpenRouter
- Vector storage with Pinecone or ChromaDB
- RESTful API endpoints

## Installation

1. Install uv: `winget install astral-sh.uv`
2. Clone the repository
3. Create virtual environment: `uv venv`
4. Install dependencies: `uv sync`

## Usage

Run the application:

```bash
uv run uvicorn app.main:app --reload
```

## API Endpoints

- `GET /`: Health check
- `POST /upload`: Upload documents
- `POST /query`: Query the RAG system