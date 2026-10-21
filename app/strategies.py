"""The three strategies the benchmark compares, behind one interface.

Each takes a question and returns {"answer", "sources", "retries"}, so the
benchmark (step 6) can loop over them without caring how each works.

Compare all three on one question:
    .venv\\Scripts\\python -m app.strategies "your question"
"""

import sys
import time

from app import cag
from app.graph import ask
from app.nodes import generate, retrieve


def plain_rag(question: str) -> dict:
    """Step 2's pipeline: search once, answer from whatever came back."""
    state = {"question": question}
    state |= retrieve(state)
    state |= generate(state)
    return {"answer": state["answer"], "sources": state["sources"], "retries": 0}


def agentic_rag(question: str) -> dict:
    """Step 3's graph: grade, rewrite and retry (max 2) before answering."""
    final = ask(question)
    return {"answer": final["answer"], "sources": final["sources"],
            "retries": final.get("attempts", 0)}


def cag_whole_paper(question: str) -> dict:
    """No search: pick 1-2 papers and put them in the prompt whole."""
    result = cag.answer(question)
    return {"answer": result["answer"], "sources": result["sources"], "retries": 0}


STRATEGIES = {
    "plain_rag": plain_rag,
    "agentic_rag": agentic_rag,
    "cag": cag_whole_paper,
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m app.strategies "your question"')
    question = " ".join(sys.argv[1:])
    for name, run in STRATEGIES.items():
        started = time.perf_counter()
        out = run(question)
        print(f"=== {name}  ({time.perf_counter() - started:.1f}s, "
              f"retries={out['retries']})")
        print(out["answer"].strip())
        print("Sources:", ", ".join(out["sources"]) or "none", "\n")
