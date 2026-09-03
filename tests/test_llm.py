import pytest

from rag_support.config import ProviderMode, Settings
from rag_support.llm.base import HistoryTurn
from rag_support.llm.factory import get_llm_provider
from rag_support.llm.offline_llm import ExtractiveLLM
from rag_support.llm.prompts import build_chat_messages, build_flat_prompt


def test_extractive_llm_returns_top_passage_with_disclaimer():
    llm = ExtractiveLLM()
    answer = llm.generate("How do I reset my password?", ["Click Forgot password."])
    assert "Click Forgot password." in answer
    assert "offline extractive fallback" in answer


def test_extractive_llm_combines_multiple_passages():
    llm = ExtractiveLLM(max_passages=2)
    answer = llm.generate("q", ["passage one", "passage two", "passage three"])
    assert "passage one" in answer
    assert "passage two" in answer
    assert "passage three" not in answer  # respects max_passages


def test_extractive_llm_handles_no_context():
    llm = ExtractiveLLM()
    answer = llm.generate("anything", [])
    assert "support@northwindcloud.com" in answer


def test_build_chat_messages_includes_system_context_and_history():
    history = [
        HistoryTurn(role="user", content="hi"),
        HistoryTurn(role="assistant", content="hello"),
    ]
    messages = build_chat_messages("What now?", ["[ctx]"], history)
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello"}
    assert "What now?" in messages[-1]["content"]
    assert "[1]" in messages[-1]["content"] or "[ctx]" in messages[-1]["content"]


def test_build_flat_prompt_numbers_context_passages():
    prompt = build_flat_prompt("q", ["first passage", "second passage"])
    assert "[1] first passage" in prompt
    assert "[2] second passage" in prompt
    assert prompt.strip().endswith("Answer:")


def test_llm_factory_falls_back_to_offline(monkeypatch):
    monkeypatch.setattr("rag_support.llm.factory._openai_available", lambda settings: False)
    monkeypatch.setattr("rag_support.llm.factory._local_available", lambda: False)
    settings = Settings(llm_provider=ProviderMode.AUTO, openai_api_key=None)
    provider = get_llm_provider(settings)
    assert provider.name == "offline"


def test_explicit_local_mode_without_extra_installed_raises(monkeypatch):
    monkeypatch.setattr("rag_support.llm.factory._local_available", lambda: False)
    settings = Settings(llm_provider=ProviderMode.LOCAL)
    with pytest.raises(RuntimeError, match="local-ml"):
        get_llm_provider(settings)
