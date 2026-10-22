"""The main -> fallback model chain. No API calls: fake models stand in."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from app import nodes


def test_llm_tries_the_main_model_then_the_fallback():
    chain = nodes.llm()
    assert chain.runnable.model == nodes.LLM_MODEL
    assert chain.runnable.timeout == nodes.PRIMARY_TIMEOUT_S
    assert chain.runnable.max_retries == 0          # fail fast to the fallback
    assert [f.model for f in chain.fallbacks] == [nodes.FALLBACK_MODEL]


def test_primary_timeout_is_shorter_than_fallback():
    assert nodes.PRIMARY_TIMEOUT_S < nodes.LLM_TIMEOUT_S


def test_a_failing_main_model_is_rescued_by_the_fallback():
    """The behaviour we rely on, with stand-ins for the two Gemini models."""
    def overloaded(_):
        raise TimeoutError("main model took longer than PRIMARY_TIMEOUT_S")
    backup = FakeListChatModel(responses=["answer from the fallback"])
    chain = RunnableLambda(overloaded).with_fallbacks([backup])
    assert chain.invoke("question").content == "answer from the fallback"


def test_when_both_fail_the_error_still_surfaces():
    def overloaded(_):
        raise TimeoutError("down")
    chain = RunnableLambda(overloaded).with_fallbacks([RunnableLambda(overloaded)])
    with pytest.raises(TimeoutError):
        chain.invoke("question")
