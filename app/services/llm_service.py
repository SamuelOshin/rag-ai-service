from openai import AsyncOpenAI
from app.core.config import get_settings
import httpx

settings = get_settings()

class OpenRouterService:
    """
    Handles all interactions with the LLM provider (OpenRouter).
    Uses the standard OpenAI SDK which is compatible with OpenRouter.
    """
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url=settings.OPENROUTER_BASE_URL,
            # OpenRouter specific headers
            default_headers={
                "HTTP-Referer": "https://localhost:8000", 
                "X-Title": settings.PROJECT_NAME, 
            }
        )

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generates vector embeddings for a given text.
        """
        try:
            # Note: Ensure the model selected in config supports embeddings
            response = await self.client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding Error: {e}")
            raise e

    async def generate_answer(self, context: str, question: str) -> str:
        """
        Generates a RAG response based on context and user question.
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
            temperature=0.3  # Lower temperature for more factual answers
        )
        return response.choices[0].message.content