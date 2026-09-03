"""Zero-dependency, zero-download embedding tier.

Neither `openai` nor a Hugging Face model are always available -- an API key
might not be configured, or the environment might have no outbound network
access at all (true of the sandbox this project was built in; see
docs/data_pipeline.md). `OfflineEmbedder` exists so the pipeline is *always*
runnable end-to-end: it uses scikit-learn's `HashingVectorizer`, a purely
local, stateless bag-of-words-with-hashing technique.

This is a real, documented trade-off, not a hidden shortcut: hashing
embeddings capture lexical (word-overlap) similarity, not the semantic
similarity a transformer model gives you, so retrieval quality is lower.
It is the correct default for CI, tests, and offline demos, and the wrong
choice for a real deployment -- set EMBEDDING_PROVIDER=openai or `local`
there.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from rag_support.embeddings.base import EmbeddingProvider
from rag_support.logging_config import get_logger

logger = get_logger(__name__)


class OfflineEmbedder(EmbeddingProvider):
    name = "offline"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        # alternate_sign=True + l2 norm makes the hashed features behave
        # reasonably like a normalized embedding for cosine similarity.
        # ngram_range=(1, 2) gives it a little phrase-level signal beyond
        # pure unigram overlap.
        self._vectorizer = HashingVectorizer(
            n_features=dim, alternate_sign=True, norm="l2", ngram_range=(1, 2)
        )
        logger.warning(
            "Using the offline hashing embedder -- lexical similarity only, "
            "not a semantic model. Set EMBEDDING_PROVIDER=openai or =local "
            "for production-quality retrieval."
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self._vectorizer.transform(texts)
        dense: np.ndarray = matrix.toarray()
        return dense.tolist()
