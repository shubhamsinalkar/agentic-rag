"""The shared state that flows through the graph.

Lives in its own file because both graph.py (wiring) and nodes.py (work)
need it; putting it in graph.py would make the two files import each other.

total=False: a node only returns the keys it changed, and a fresh run starts
with just {"question": ...}, so nodes read optional keys with .get().
"""

from typing import TypedDict

from langchain_core.documents import Document


class State(TypedDict, total=False):
    question: str               # the user's question, never changed
    route: str                  # router: factual / conceptual / comparison / off_topic
    search_query: str          # what we search with; rewritten on retries
    documents: list[Document]   # chunks to grade; after grade: relevant only
    seen_ids: list[str]         # every chunk already graded, so retries
                                # look at NEW chunks instead of the same 4
    sufficient: bool            # grader: do the relevant chunks answer it?
    missing: str                # grader: what info is still missing
    attempts: int               # rewrites done so far; caps the retry loop
    answer: str                 # written by generate
    sources: list[str]          # "paper.pdf p.N" for chunks the answer cites
