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
    ("as shown [chexnet.pdf p.5].", ["chexnet.pdf p.5"]),
    ("as shown [chexnet.pdf p. 5].", ["chexnet.pdf p.5"]),
    ("[chexnet.pdf p.2, p.4]", ["chexnet.pdf p.2", "chexnet.pdf p.4"]),
    ("[planet_nine.pdf p.3]", []),                                  # not a paper we gave it
    ("[chexnet.pdf p.5] [chexnet.pdf p.5]", ["chexnet.pdf p.5"]),
])
def test_cited_pages(text, expected):
    assert cag.cited_pages(text, ["chexnet.pdf"]) == expected


def test_every_catalog_entry_is_a_real_file():
    for name in cag.PAPERS:
        assert (cag.DOCS_DIR / name).exists(), name


@needs_api
def test_selects_the_right_paper():
    assert cag.select_papers("How did the CheXNet authors decide that CheXNet "
                             "beats radiologists?") == ["chexnet.pdf"]


@needs_api
def test_selects_both_papers_for_a_comparison():
    assert cag.select_papers("How do CheXNet and the Transformer differ in their "
                             "Adam optimizer settings?") == ["chexnet.pdf", "transformer.pdf"]


@needs_api
def test_selects_nothing_for_an_off_topic_question():
    assert cag.select_papers("Who won the 2022 FIFA World Cup?") == []
