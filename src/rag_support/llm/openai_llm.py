"""OpenAI chat-completion backend -- the default (highest quality) tier."""

from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_support.llm.base import HistoryTurn, LLMProvider
from rag_support.llm.prompts import build_chat_messages
from rag_support.logging_config import get_logger

logger = get_logger(__name__)


class OpenAILLM(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, temperature: float = 0.2) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(Exception),
    )
    def generate(
        self, question: str, context: list[str], history: list[HistoryTurn] | None = None
    ) -> str:
        messages = build_chat_messages(question, context, history)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self._temperature,
            max_tokens=600,
        )
        return (response.choices[0].message.content or "").strip()
