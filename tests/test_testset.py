"""The benchmark is only as good as its test set: check it's well-formed."""

import json
from collections import Counter

from app.ingest import DOCS_DIR, ROOT

TYPES = {"factual", "conceptual", "comparison", "out_of_scope"}


def test_testset_is_well_formed():
    items = json.loads((ROOT / "data" / "testset.json").read_text(encoding="utf-8"))
    assert len(items) == 30
    assert len({q["id"] for q in items}) == 30, "ids must be unique"
    for q in items:
        assert q["type"] in TYPES, q["id"]
        assert q["question"].strip() and q["expected_answer"].strip(), q["id"]
        for paper in q["papers"]:
            assert (DOCS_DIR / paper).exists(), (q["id"], paper)
        # answerable questions must say where the answer is
        assert bool(q["papers"]) == (q["type"] != "out_of_scope"), q["id"]
    counts = Counter(q["type"] for q in items)
    assert all(counts[t] >= 2 for t in TYPES), counts
