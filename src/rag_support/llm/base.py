"""The `LLMProvider` contract every generation backend implements.

The signature is deliberately structured (question + ranked context chunks +
optional history) rather than a single opaque prompt string: every backend,
including the offline extractive fallback, gets the same well-typed inputs
and is free to format them however suits it (a chat `messages` array for
OpenAI, a flattened instruction string for a local seq2seq model, or a
straight lookup for the extractive tier).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class HistoryTurn:
    role: str  # "user" | "assistant"
    content: str


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        question: str,
        context: list[str],
        history: list[HistoryTurn] | None = None,
    ) -> str:
        """Produce an answer grounded in `context` (ranked, most relevant first)."""
        raise NotImplementedError
