"""Zero-dependency, zero-download generation tier.

Same rationale as `embeddings.offline_embedder`: this environment (and CI)
can't always reach an LLM API or download model weights, so the pipeline
needs a fallback that always works. `ExtractiveLLM` doesn't generate free
text at all -- it returns the best-matching retrieved passage(s), verbatim,
with a clear disclaimer. That makes it honest rather than impressive: it's
there so the system is demoable offline, not as the recommended answer path.
"""

from __future__ import annotations

from rag_support.llm.base import HistoryTurn, LLMProvider
from rag_support.logging_config import get_logger

logger = get_logger(__name__)

_DISCLAIMER = (
    "\n\n_(This answer was returned by the offline extractive fallback -- the "
    "single most relevant passage from the knowledge base, with no language "
    "model involved. Configure LLM_PROVIDER=openai or =local for a fully "
    "generated, conversational answer.)_"
)


class ExtractiveLLM(LLMProvider):
    name = "offline"

    def __init__(self, max_passages: int = 2) -> None:
        self.max_passages = max_passages
        logger.warning(
            "Using the offline extractive fallback -- no language model is "
            "involved, answers are the top retrieved passage(s) verbatim."
        )

    def generate(
        self, question: str, context: list[str], history: list[HistoryTurn] | None = None
    ) -> str:
        if not context:
            return (
                "I couldn't find anything in the knowledge base about that. "
                "Please contact support@northwindcloud.com for help." + _DISCLAIMER
            )

        passages = context[: self.max_passages]
        if len(passages) == 1:
            body = passages[0]
        else:
            body = "\n\n".join(f"- {p}" for p in passages)
        return f"{body}{_DISCLAIMER}"
