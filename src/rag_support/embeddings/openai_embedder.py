"""OpenAI embeddings backend -- the default (highest quality) tier."""

from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_support.embeddings.base import EmbeddingProvider
from rag_support.logging_config import get_logger

logger = get_logger(__name__)

# Batch conservatively: OpenAI's embeddings endpoint accepts large batches,
# but keeping requests modest bounds both worst-case latency of a single
# call and the blast radius of a single retry.
_BATCH_SIZE = 96


class OpenAIEmbedder(EmbeddingProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, dim: int) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=batch)
        return [item.embedding for item in response.data]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            logger.debug("Embedding batch %d-%d via OpenAI (%s)", i, i + len(batch), self._model)
            results.extend(self._embed_batch(batch))
        return results
