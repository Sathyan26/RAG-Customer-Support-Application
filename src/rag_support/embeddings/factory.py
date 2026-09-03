"""Resolve `EMBEDDING_PROVIDER` (auto|openai|local|offline) to a concrete
`EmbeddingProvider`, with AUTO gracefully degrading through the tiers.

This is the one place that knows the fallback order -- callers (the
pipeline, the CLI, tests) just ask for "the configured embedder" and get
something that works.
"""

from __future__ import annotations

from importlib.util import find_spec

from rag_support.config import ProviderMode, Settings
from rag_support.embeddings.base import EmbeddingProvider
from rag_support.logging_config import get_logger

logger = get_logger(__name__)


def _openai_available(settings: Settings) -> bool:
    return bool(settings.openai_api_key) and find_spec("openai") is not None


def _local_available() -> bool:
    return find_spec("sentence_transformers") is not None


def build_openai_embedder(settings: Settings) -> EmbeddingProvider:
    from rag_support.embeddings.openai_embedder import OpenAIEmbedder

    if not settings.openai_api_key:
        raise RuntimeError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set")
    return OpenAIEmbedder(
        api_key=settings.openai_api_key,
        model=settings.openai_embedding_model,
        dim=settings.openai_embedding_dim,
    )


def build_local_embedder(settings: Settings) -> EmbeddingProvider:
    from rag_support.embeddings.local_embedder import LocalEmbedder

    if not _local_available():
        raise RuntimeError(
            "EMBEDDING_PROVIDER=local requires the local-ml extra: "
            "pip install -e '.[local-ml]'"
        )
    return LocalEmbedder(model_name=settings.local_embedding_model, dim=settings.local_embedding_dim)


def build_offline_embedder(settings: Settings) -> EmbeddingProvider:
    from rag_support.embeddings.offline_embedder import OfflineEmbedder

    return OfflineEmbedder(dim=settings.offline_embedding_dim)


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    mode = settings.embedding_provider

    if mode == ProviderMode.OPENAI:
        return build_openai_embedder(settings)
    if mode == ProviderMode.LOCAL:
        return build_local_embedder(settings)
    if mode == ProviderMode.OFFLINE:
        return build_offline_embedder(settings)

    # AUTO: best available tier, degrading gracefully and logging why.
    if _openai_available(settings):
        logger.info("EMBEDDING_PROVIDER=auto -> openai (API key configured)")
        return build_openai_embedder(settings)
    if _local_available():
        logger.info("EMBEDDING_PROVIDER=auto -> local (sentence-transformers installed)")
        return build_local_embedder(settings)
    logger.info("EMBEDDING_PROVIDER=auto -> offline (no OpenAI key, no local-ml extra installed)")
    return build_offline_embedder(settings)
