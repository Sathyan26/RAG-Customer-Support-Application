"""Resolve `LLM_PROVIDER` (auto|openai|local|offline) to a concrete
`LLMProvider`, mirroring `embeddings.factory`'s fallback logic exactly."""

from __future__ import annotations

from importlib.util import find_spec

from rag_support.config import ProviderMode, Settings
from rag_support.llm.base import LLMProvider
from rag_support.logging_config import get_logger

logger = get_logger(__name__)


def _openai_available(settings: Settings) -> bool:
    return bool(settings.openai_api_key) and find_spec("openai") is not None


def _local_available() -> bool:
    return find_spec("transformers") is not None


def build_openai_llm(settings: Settings) -> LLMProvider:
    from rag_support.llm.openai_llm import OpenAILLM

    if not settings.openai_api_key:
        raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set")
    return OpenAILLM(api_key=settings.openai_api_key, model=settings.openai_chat_model)


def build_local_llm(settings: Settings) -> LLMProvider:
    from rag_support.llm.local_llm import LocalLLM

    if not _local_available():
        raise RuntimeError(
            "LLM_PROVIDER=local requires the local-ml extra: pip install -e '.[local-ml]'"
        )
    return LocalLLM(model_name=settings.local_llm_model)


def build_offline_llm(settings: Settings) -> LLMProvider:  # noqa: ARG001 - uniform signature
    from rag_support.llm.offline_llm import ExtractiveLLM

    return ExtractiveLLM()


def get_llm_provider(settings: Settings) -> LLMProvider:
    mode = settings.llm_provider

    if mode == ProviderMode.OPENAI:
        return build_openai_llm(settings)
    if mode == ProviderMode.LOCAL:
        return build_local_llm(settings)
    if mode == ProviderMode.OFFLINE:
        return build_offline_llm(settings)

    if _openai_available(settings):
        logger.info("LLM_PROVIDER=auto -> openai (API key configured)")
        return build_openai_llm(settings)
    if _local_available():
        logger.info("LLM_PROVIDER=auto -> local (transformers installed)")
        return build_local_llm(settings)
    logger.info("LLM_PROVIDER=auto -> offline (no OpenAI key, no local-ml extra installed)")
    return build_offline_llm(settings)
