"""Local, no-API-cost embeddings backend using sentence-transformers.

Downloads the model weights from the Hugging Face Hub on first use and
caches them under the usual `~/.cache/huggingface` -- outbound network access
is required exactly once per model, not per request. See
`rag_support.embeddings.offline_embedder` for a tier that needs no
downloads at all.
"""

from __future__ import annotations

from rag_support.embeddings.base import EmbeddingProvider
from rag_support.logging_config import get_logger

logger = get_logger(__name__)


class LocalEmbedder(EmbeddingProvider):
    name = "local"

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model %s (first run downloads it)", model_name)
        self._model = SentenceTransformer(model_name)
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]
