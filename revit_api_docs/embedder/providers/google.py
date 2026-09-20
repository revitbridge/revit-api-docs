"""Google embedding provider (google-genai SDK; optional `pipeline` extra)."""
from .base import BaseEmbedding
from ...config import get_api_key


class GoogleEmbedding(BaseEmbedding):

    def __init__(self, model: str = "text-embedding-004", dimension: int = 768, api_key_env: str = "GOOGLE_API_KEY"):
        from google import genai
        self._model = model
        self._dimension = dimension
        self._client = genai.Client(api_key=get_api_key(api_key_env))

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding (the Google API accepts batches)."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=[query],
        )
        return result.embeddings[0].values
