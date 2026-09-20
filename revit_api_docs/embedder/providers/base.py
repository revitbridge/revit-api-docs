"""Abstract base for embedding providers (OpenAI-compatible, Google)."""
from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Uniform embedding interface."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name, recorded in the index meta.json."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension."""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        pass
