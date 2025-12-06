from openai import AsyncOpenAI
from app.core.config import get_settings
import httpx

settings = get_settings()

class OpenRouterService:
    """Handles all interactions with the LLM provider via OpenRouter."""

    def __init__(self):
        """Initializes the OpenAI client with OpenRouter configuration."""
        self.client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://localhost:8000", 
                "X-Title": settings.PROJECT_NAME, 
            }
        )

    async def get_embedding(self, text: str) -> list[float]:
        """Generates vector embeddings for the given text.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector.

        Raises:
            Exception: If embedding generation fails.

        Examples:
            >>> emb = await service.get_embedding("Hello world")
            >>> len(emb) > 0
            True
        """
        try:
            response = await self.client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding Error: {e}")
            raise e

    async def generate_answer(self, context: str, question: str) -> str:
        """Generates a RAG response based on context and question.

        Args:
            context (str): Retrieved context from documents.
            question (str): User's question.

        Returns:
            str: Generated answer.

        Raises:
            Exception: If answer generation fails.

        Examples:
            >>> answer = await service.generate_answer("Context text", "What is AI?")
            >>> isinstance(answer, str)
            True
        """
        system_prompt = (
            "You are an intelligent assistant. "
            "Use the provided context to answer the user's question accurately. "
            "If the answer is not in the context, say you don't know."
        )
        
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        response = await self.client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content