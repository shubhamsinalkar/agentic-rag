"""The shared state that flows through the graph.

Lives in its own file because both graph.py (wiring) and nodes.py (work)
need it; putting it in graph.py would make the two files import each other.
"""

from typing import TypedDict

from langchain_core.documents import Document


class State(TypedDict):
    question: str               # the user's question, unchanged
    documents: list[Document]   # chunks found by retrieve
    answer: str                 # written by generate
    sources: list[str]          # "paper.pdf p.N" citations for the answer
    # Added later: attempts (step 3, caps the retry loop), route (step 5).
