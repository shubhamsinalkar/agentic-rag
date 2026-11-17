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

DEMO_QUESTION = ("How did the CheXNet authors decide that CheXNet is statistically "
                 "significantly better than the radiologists?")

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
    return nodes.grade({"question": DEMO_QUESTION, "documents": docs})


# --------------------------------------------------- grader (real API) ---

@needs_api
def test_rejects_chunks_from_an_unrelated_paper():
    """Planet Nine chunks say nothing about CheXNet: all must be rejected."""
    result = _grade(_chunks({"source": "planet_nine.pdf"}))
    assert result["documents"] == []
    assert result["sufficient"] is False


@needs_api
def test_on_topic_but_vague_chunks_are_not_sufficient():
    """The trap: CheXNet's first page says it "exceeds average radiologist
    performance" and scores high in vector search, but never says how
    significance was tested."""
    result = _grade(_chunks({"$and": [{"source": "chexnet.pdf"}, {"page": 1}]}))
    assert result["sufficient"] is False
    assert result["missing"], "grader should say what is missing"


@needs_api
def test_accepts_the_chunk_that_answers():
    """Page 3 states the test (bootstrap CI of the F1 difference excludes 0)."""
    answer_chunk = _chunks({"source": "chexnet.pdf"}, limit=1,
                           contains="does not contain 0")
    assert answer_chunk, "expected chunk not found in the index"
    result = _grade(answer_chunk + _chunks({"source": "planet_nine.pdf"}, limit=2))
    assert result["sufficient"] is True
    assert [d.id for d in result["documents"]] == [answer_chunk[0].id]


@needs_api
def test_one_side_of_a_comparison_counts_as_relevant():
    """For a comparison every chunk covers only one side; the grader must
    still keep it, or comparisons can never be answered."""
    chexnet_adam = _chunks({"source": "chexnet.pdf"}, limit=1, contains="0.999")
    result = nodes.grade({
        "question": "How do the Adam optimizer settings of CheXNet and the Transformer differ?",
        "documents": chexnet_adam,
    })
    assert [d.id for d in result["documents"]] == [chexnet_adam[0].id]
    assert result["sufficient"] is False  # the Transformer side is still missing


# ------------------------------------------------- retrieve (real search) ---

@needs_api
def test_retry_searches_new_chunks_not_the_same_ones():
    first = nodes.retrieve({"question": DEMO_QUESTION})
    second = nodes.retrieve({"question": DEMO_QUESTION, "documents": [],
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
    result = nodes.generate({"question": DEMO_QUESTION, "documents": []})
    assert result == {"answer": nodes.IDK, "sources": [], "citations": []}
