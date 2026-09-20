"""Embedding provider factory: builds the provider named in config["embedding"]."""
from .base import BaseEmbedding


def create_embedding(config: dict) -> BaseEmbedding:
    """Create the embedding provider selected by the config."""
    provider = config["embedding"]["provider"]
    model_config = config["embedding"]["models"][provider]

    if provider == "google":
        from .google import GoogleEmbedding
        return GoogleEmbedding(**model_config)

    elif provider == "openai":
        from .openai import OpenAIEmbedding
        return OpenAIEmbedding(**model_config)

    elif provider == "local_hf":
        raise NotImplementedError("local HuggingFace embedding is not implemented")

    elif provider == "zhipu":
        raise NotImplementedError("zhipu embedding is not implemented")

    else:
        raise ValueError(f"unsupported embedding provider: {provider}")
