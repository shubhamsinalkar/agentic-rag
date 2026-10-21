"""CAG: citation parsing (no API) and paper selection (real API)."""

import os

import pytest

from app import cag
from app.ingest import CHROMA_DIR

needs_api = pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") and CHROMA_DIR.exists()),
    reason="needs GOOGLE_API_KEY in .env",
)


@pytest.mark.parametrize("text, expected", [
    ("as shown [corrective_rag.pdf p.5].", ["corrective_rag.pdf p.5"]),
    ("as shown [corrective_rag.pdf p. 5].", ["corrective_rag.pdf p.5"]),
    ("[corrective_rag.pdf p.2, p.4]", ["corrective_rag.pdf p.2", "corrective_rag.pdf p.4"]),
    ("[react.pdf p.3]", []),                                  # not a paper we gave it
    ("[corrective_rag.pdf p.5] [corrective_rag.pdf p.5]", ["corrective_rag.pdf p.5"]),
])
def test_cited_pages(text, expected):
    assert cag.cited_pages(text, ["corrective_rag.pdf"]) == expected


def test_every_catalog_entry_is_a_real_file():
    for name in cag.PAPERS:
        assert (cag.DOCS_DIR / name).exists(), name


@needs_api
def test_selects_the_right_paper():
    assert cag.select_papers("How does CRAG decide a retrieved document is "
                             "incorrect?") == ["corrective_rag.pdf"]


@needs_api
def test_selects_both_papers_for_a_comparison():
    assert cag.select_papers("How do CRAG and Self-RAG differ in judging "
                             "retrieved documents?") == ["corrective_rag.pdf", "self_rag.pdf"]


@needs_api
def test_selects_nothing_for_an_off_topic_question():
    assert cag.select_papers("Who won the 2022 FIFA World Cup?") == []
