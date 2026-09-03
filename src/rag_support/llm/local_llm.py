"""Local, no-API-cost generation backend using a HuggingFace seq2seq model.

Defaults to google/flan-t5-base: small enough to run on CPU, instruction-
tuned, and good enough for grounded, short support answers. Swap
LOCAL_LLM_MODEL for anything else `transformers` can load as
text2text-generation or text-generation.
"""

from __future__ import annotations

from rag_support.llm.base import HistoryTurn, LLMProvider
from rag_support.llm.prompts import build_flat_prompt
from rag_support.logging_config import get_logger

logger = get_logger(__name__)


class LocalLLM(LLMProvider):
    name = "local"

    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        logger.info("Loading local LLM %s (first run downloads it)", model_name)
        self._pipe = pipeline("text2text-generation", model=model_name)

    def generate(
        self, question: str, context: list[str], history: list[HistoryTurn] | None = None
    ) -> str:
        prompt = build_flat_prompt(question, context, history)
        # flan-t5's context window is modest; truncate defensively rather
        # than letting the underlying tokenizer error out on long context.
        output = self._pipe(prompt, max_new_tokens=256, truncation=True)
        return output[0]["generated_text"].strip()
