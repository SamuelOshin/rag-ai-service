import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import get_settings
from typing import List, Dict

settings = get_settings()

class VectorDBService:
    """
    Abstraction layer for Vector Database.
    Currently implements ChromaDB via HTTP (Client/Server mode).
    """
    def __init__(self):
        # Connect to the Chroma container
        self.client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT
        )
        self.collection = self.client.get_or_create_collection(name=settings.COLLECTION_NAME)

    def upsert_chunks(self, doc_id: int, chunks: List[str], embeddings: List[List[float]]):
        """
        Stores chunks and their embeddings.
        ID Format: {doc_id}_{chunk_index}
        """
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"doc_id": str(doc_id), "text": chunk} for chunk in chunks]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_vector: List[float], k: int = 3) -> List[Dict]:
        """
        Performs semantic search using the query vector.
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        output = []
        if results['metadatas'] and results['distances']:
            # Chroma returns a list of lists (one per query)
            metas = results['metadatas'][0]
            dists = results['distances'][0]
            
            for i, meta in enumerate(metas):
                output.append({
                    "text": meta["text"],
                    "doc_id": int(meta["doc_id"]),
                    "score": 1.0 - dists[i]  # Convert distance to similarity score approx
                })
        
        return output