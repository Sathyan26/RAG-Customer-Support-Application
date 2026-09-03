"""Prompt construction shared by the OpenAI, local, and offline providers.

Kept separate from any one provider so the grounding/citation instructions
are defined exactly once.
"""

from __future__ import annotations

from rag_support.llm.base import HistoryTurn

SYSTEM_PROMPT = (
    "You are the customer support assistant for Northwind Cloud. Answer the "
    "customer's question using ONLY the numbered context passages provided -- "
    "do not use outside knowledge and do not make anything up. Cite the "
    "passage(s) you used with bracketed numbers like [1] or [2]. If the "
    "context does not contain enough information to answer, say so plainly "
    "and suggest the customer contact support@northwindcloud.com. Keep "
    "answers concise and friendly."
)


def format_context(context: list[str]) -> str:
    return "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(context))


def build_chat_messages(
    question: str, context: list[str], history: list[HistoryTurn] | None = None
) -> list[dict[str, str]]:
    """Messages array for OpenAI-style chat completion APIs."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": turn.role, "content": turn.content})
    user_content = f"Context:\n{format_context(context)}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})
    return messages


def build_flat_prompt(
    question: str, context: list[str], history: list[HistoryTurn] | None = None
) -> str:
    """Single flattened instruction string, for models without a chat API
    (local seq2seq/causal models)."""
    parts = [SYSTEM_PROMPT, "", f"Context:\n{format_context(context)}"]
    if history:
        parts.append("")
        parts.append("Conversation so far:")
        parts.extend(f"{turn.role}: {turn.content}" for turn in history)
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("Answer:")
    return "\n".join(parts)
