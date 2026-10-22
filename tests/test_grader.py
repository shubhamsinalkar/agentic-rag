"""Does the correction loop actually work?

If the grader says "relevant" to everything, the loop never fires and the
project is plain RAG with extra steps. These tests feed it chunks we KNOW
are wrong, chunks that are on-topic but don't answer, and the chunk that
does answer.

The grader tests call the real Gemini API (free tier) and need the index:
    .venv\\Scripts\\python -m app.ingest      (once)
    .venv\\Scripts\\python -m pytest -v
"""

import os

import pytest
from langchain_core.documents import Document

from app import nodes
from app.ingest import CHROMA_DIR

CRAG_QUESTION = "How does CRAG decide a retrieved document is incorrect?"

needs_api = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and CHROMA_DIR.exists()),
    reason="needs GOOGLE_API_KEY in .env and a built index",
)


def _chunks(where: dict, limit: int = 4, contains: str | None = None) -> list[Document]:
    """Pull specific chunks straight from Chroma by metadata, no search."""
    got = nodes._vectorstore().get(where=where, include=["documents", "metadatas"])
    docs = [Document(page_content=text, metadata=meta, id=chunk_id)
            for chunk_id, text, meta in zip(got["ids"], got["documents"], got["metadatas"])]
    if contains:
        docs = [d for d in docs if contains in d.page_content]
    return docs[:limit]


def _grade(docs: list[Document]) -> dict:
    return nodes.grade({"question": CRAG_QUESTION, "documents": docs})


# --------------------------------------------------- grader (real API) ---

@needs_api
def test_rejects_chunks_from_an_unrelated_paper():
    """ReAct chunks say nothing about CRAG: all must be rejected."""
    result = _grade(_chunks({"source": "react.pdf"}))
    assert result["documents"] == []
    assert result["sufficient"] is False


@needs_api
def test_on_topic_but_vague_chunks_are_not_sufficient():
    """The trap: CRAG's title page and abstract are about the right topic
    and score high in vector search, but don't contain the rule."""
    result = _grade(_chunks({"$and": [{"source": "corrective_rag.pdf"},
                                      {"page": {"$in": [1, 10]}}]}))
    assert result["sufficient"] is False
    assert result["missing"], "grader should say what is missing"


@needs_api
def test_first_search_for_the_demo_question_is_not_sufficient():
    """Found live in step 7: these exact chunks (CRAG overview pages) were
    once graded sufficient, so the loop never fired and the answer was vague.
    They name the Correct/Incorrect/Ambiguous actions but not the rule."""
    docs = nodes.retrieve({"question": CRAG_QUESTION})["documents"]
    assert _grade(docs)["sufficient"] is False


@needs_api
def test_accepts_the_chunk_that_answers():
    """Page 5 states the rule (all scores below the lower threshold)."""
    answer_chunk = _chunks({"source": "corrective_rag.pdf"}, limit=1,
                           contains="below the lower threshold")
    assert answer_chunk, "expected chunk not found in the index"
    result = _grade(answer_chunk + _chunks({"source": "react.pdf"}, limit=2))
    assert result["sufficient"] is True
    assert [d.id for d in result["documents"]] == [answer_chunk[0].id]


@needs_api
def test_one_side_of_a_comparison_counts_as_relevant():
    """Bug found in step 5: for "how does ReAct differ from RAG?", every
    chunk covers only one side, and the grader used to reject them all."""
    react_actions = _chunks({"source": "react.pdf"}, limit=1, contains="lookup[string]")
    result = nodes.grade({
        "question": "How does the way ReAct retrieves information differ from the way RAG retrieves it?",
        "documents": react_actions,
    })
    assert [d.id for d in result["documents"]] == [react_actions[0].id]
    assert result["sufficient"] is False  # the RAG side is still missing


# ------------------------------------------------- retrieve (real search) ---

@needs_api
def test_retry_searches_new_chunks_not_the_same_ones():
    first = nodes.retrieve({"question": CRAG_QUESTION})
    second = nodes.retrieve({"question": CRAG_QUESTION, "documents": [],
                             "seen_ids": first["seen_ids"]})
    assert not {d.id for d in first["documents"]} & {d.id for d in second["documents"]}
    assert len(second["documents"]) == 4


# ----------------------------------------- the branch (pure logic, no API) ---

@pytest.mark.parametrize("sufficient, attempts, expected", [
    (True, 0, "generate"),    # good evidence: answer now
    (False, 0, "rewrite"),    # bad evidence, retries left: try again
    (False, 1, "rewrite"),
    (False, 2, "generate"),   # cap reached: stop looping, answer or say IDK
    (False, 99, "generate"),
])
def test_decide_after_grade(sufficient, attempts, expected):
    state = {"sufficient": sufficient, "attempts": attempts}
    assert nodes.decide_after_grade(state) == expected


def test_generate_skips_llm_when_nothing_relevant():
    result = nodes.generate({"question": CRAG_QUESTION, "documents": []})
    assert result == {"answer": nodes.IDK, "sources": []}
