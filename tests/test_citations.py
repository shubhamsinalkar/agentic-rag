"""Cited passages returned to the UI. No API calls."""

from langchain_core.documents import Document

from app import nodes


def _doc(page, text):
    return Document(page_content=text, metadata={"source": "corrective_rag.pdf", "page": page})


def test_only_cited_blocks_come_back_in_first_cited_order():
    docs = [_doc(3, "overview"), _doc(5, "Incorrect if below\nthe lower threshold"), _doc(6, "x")]
    out = nodes._citations("Rule is X [2]. Also [1] and again [2]. Bad ref [9].", docs)
    assert [c["n"] for c in out] == [2, 1]
    assert out[0] == {"n": 2, "source": "corrective_rag.pdf", "page": 5,
                      "text": "Incorrect if below the lower threshold"}


def test_long_passages_are_trimmed_to_a_quotation():
    out = nodes._citations("[1]", [_doc(1, "word " * 400)])
    assert len(out[0]["text"]) == nodes.QUOTE_CHARS + 1 and out[0]["text"].endswith("…")
