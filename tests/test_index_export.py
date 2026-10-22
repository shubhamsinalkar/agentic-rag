"""The shipped index (data/index) must match the PDFs exactly. No API calls.

If a pypdf or splitter upgrade changes the extracted text, chunk hashes stop
matching and a deploy would fail; this test catches that first.
"""

import json

import numpy as np

from app.corpus import CORPUS
from app.ingest import (EMBED_DIMS, INDEX_EXPORT, _text_hash, chunk_id, load_pages, split)


def test_export_matches_freshly_split_pdfs():
    manifest = json.loads((INDEX_EXPORT / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(INDEX_EXPORT / "embeddings.npy")
    assert vectors.shape == (len(manifest["chunks"]), EMBED_DIMS)

    saved = {c["id"]: c["sha256"] for c in manifest["chunks"]}
    fresh = {chunk_id(d): _text_hash(d.page_content) for d in split(load_pages())}
    assert fresh == saved


def test_chunk_ids_are_unique_and_readable():
    ids = [chunk_id(d) for d in split(load_pages())]
    assert len(ids) == len(set(ids))
    paper, page, offset = ids[0].split(":")
    assert paper.endswith(".pdf") and page.isdigit() and offset.isdigit()


def test_corpus_lists_every_local_paper():
    from app.ingest import DOCS_DIR
    assert {p.file for p in CORPUS} == {p.name for p in DOCS_DIR.glob("*.pdf")}
