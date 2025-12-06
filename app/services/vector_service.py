import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import get_settings
from typing import List, Dict

settings = get_settings()

class VectorDBService:
    """Abstraction layer for vector database operations using ChromaDB."""

    def __init__(self):
        """Initializes the ChromaDB client and collection."""
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT
        )
        self.collection = self.client.get_or_create_collection(name=settings.COLLECTION_NAME)

    def upsert_chunks(self, doc_id: int, chunks: List[str], embeddings: List[List[float]]):
        """Stores text chunks and their embeddings in the vector database.

        Args:
            doc_id (int): ID of the document.
            chunks (List[str]): List of text chunks.
            embeddings (List[List[float]]): Corresponding embeddings.

        Examples:
            >>> service.upsert_chunks(1, ["chunk1", "chunk2"], [[0.1, 0.2], [0.3, 0.4]])
            # Stores chunks in vector DB
        """
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": str(doc_id), "text": chunk} for chunk in chunks]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_vector: List[float], k: int = 3) -> List[Dict]:
        """Performs semantic search using the query vector.

        Args:
            query_vector (List[float]): Embedding of the query.
            k (int): Number of results to return.

        Returns:
            List[Dict]: List of matching chunks with text, doc_id, and score.

        Examples:
            >>> results = service.search([0.1, 0.2, 0.3], k=5)
            >>> len(results) <= 5
            True
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        output = []
        if results['metadatas'] and results['distances']:
            metas = results['metadatas'][0]
            dists = results['distances'][0]
            
            for i, meta in enumerate(metas):
                output.append({
                    "text": meta["text"],
                    "doc_id": int(meta["doc_id"]),
                    "score": 1.0 - dists[i]
                })
        
        return output