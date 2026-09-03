import numpy as np
import pytest

from rag_support.config import ProviderMode, Settings
from rag_support.embeddings.factory import get_embedding_provider
from rag_support.embeddings.offline_embedder import OfflineEmbedder


def test_offline_embedder_returns_vectors_of_the_configured_dimension():
    embedder = OfflineEmbedder(dim=128)
    vectors = embedder.embed(["reset my password", "cancel my subscription"])
    assert len(vectors) == 2
    assert all(len(v) == 128 for v in vectors)


def test_offline_embedder_handles_empty_input():
    embedder = OfflineEmbedder(dim=64)
    assert embedder.embed([]) == []


def test_offline_embedder_embed_one_matches_first_of_embed():
    embedder = OfflineEmbedder(dim=64)
    assert embedder.embed_one("hello") == embedder.embed(["hello"])[0]


def test_offline_embedder_is_deterministic():
    embedder = OfflineEmbedder(dim=64)
    a = embedder.embed(["reset my password"])[0]
    b = embedder.embed(["reset my password"])[0]
    assert a == b


def test_offline_embedder_puts_similar_text_closer_than_unrelated_text():
    embedder = OfflineEmbedder(dim=256)
    query = embedder.embed_one("how do I reset my password")
    close = embedder.embed_one("I forgot my password, how do I reset it")
    far = embedder.embed_one("do you ship internationally")

    def cosine(a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9))

    assert cosine(query, close) > cosine(query, far)


def test_factory_falls_back_to_offline_when_nothing_else_is_configured(monkeypatch):
    monkeypatch.setattr("rag_support.embeddings.factory._openai_available", lambda settings: False)
    monkeypatch.setattr("rag_support.embeddings.factory._local_available", lambda: False)
    settings = Settings(embedding_provider=ProviderMode.AUTO, openai_api_key=None)
    provider = get_embedding_provider(settings)
    assert provider.name == "offline"


def test_factory_prefers_openai_when_available(monkeypatch):
    monkeypatch.setattr("rag_support.embeddings.factory._openai_available", lambda settings: True)
    built = {}

    def fake_build_openai(settings):
        built["called"] = True

        class _Fake:
            name = "openai"

        return _Fake()

    monkeypatch.setattr("rag_support.embeddings.factory.build_openai_embedder", fake_build_openai)
    settings = Settings(embedding_provider=ProviderMode.AUTO, openai_api_key="sk-fake")
    provider = get_embedding_provider(settings)
    assert provider.name == "openai"
    assert built["called"]


def test_explicit_openai_mode_without_api_key_raises():
    settings = Settings(embedding_provider=ProviderMode.OPENAI, openai_api_key=None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        get_embedding_provider(settings)
