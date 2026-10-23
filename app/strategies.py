"""The three strategies the benchmark compares, behind one interface.

Each takes a question and returns:
    answer, sources, retries   - what the benchmark scores
    steps                      - what happened, in order, with timings
    citations                  - cited passages with their text (RAG only)
    route                      - the router's label (agentic only)

Compare all three on one question:
    .venv\\Scripts\\python -m app.strategies "your question"
"""

import sys
import time

from app import cag
from app.graph import RECURSION_LIMIT, app
from app.nodes import generate, retrieve


def _ms(since: float) -> int:
    return round((time.perf_counter() - since) * 1000)


def plain_rag(question: str) -> dict:
    """The baseline pipeline: search once, answer from whatever came back."""
    state, steps = {"question": question}, []
    t = time.perf_counter()
    state |= retrieve(state)
    steps.append({"step": "search", "query": question,
                  "found": len(state["documents"]), "ms": _ms(t)})
    t = time.perf_counter()
    state |= generate(state)
    steps.append({"step": "answer", "cited": len(state["sources"]), "ms": _ms(t)})
    return {"answer": state["answer"], "sources": state["sources"], "retries": 0,
            "steps": steps, "citations": state["citations"]}


def _describe(node: str, out: dict, seen_before: int, ms: int) -> dict:
    """Turn one graph update into a plain-language step for the UI."""
    if node == "route":
        return {"step": "route", "label": out["route"], "ms": ms}
    if node == "retrieve":
        return {"step": "search", "query": out["search_query"],
                "found": len(out["seen_ids"]) - seen_before, "ms": ms}
    if node == "grade":
        return {"step": "check", "sufficient": out["sufficient"],
                "relevant": len(out["documents"]),
                "missing": "" if out["sufficient"] else out["missing"], "ms": ms}
    if node == "rewrite":
        return {"step": "rewrite", "query": out["search_query"], "ms": ms}
    return {"step": "answer", "cited": len(out.get("sources", [])), "ms": ms}


def agentic_rag(question: str) -> dict:
    """The agent graph: grade, rewrite and retry (max 2) before answering.

    Streams the graph instead of invoke() so every step can be reported.
    """
    final, steps, seen_before = {"question": question}, [], 0
    t = time.perf_counter()
    # stream_mode="updates" yields {node_name: what that node returned}
    for update in app.stream({"question": question}, stream_mode="updates",
                             config={"recursion_limit": RECURSION_LIMIT}):
        for node, out in update.items():
            steps.append(_describe(node, out, seen_before, _ms(t)))
            t = time.perf_counter()
            final |= out
            if node == "retrieve":
                seen_before = len(out["seen_ids"])
    return {"answer": final["answer"], "sources": final["sources"],
            "retries": final.get("attempts", 0), "route": final.get("route"),
            "steps": steps, "citations": final.get("citations", [])}


def cag_whole_paper(question: str) -> dict:
    """No search: pick 1-2 papers and put them in the prompt whole."""
    t = time.perf_counter()
    papers = cag.select_papers(question)
    steps = [{"step": "pick", "papers": papers, "ms": _ms(t)}]
    t = time.perf_counter()
    result = cag.answer_from(question, papers)
    steps.append({"step": "answer", "cited": len(result["sources"]), "ms": _ms(t)})
    return {"answer": result["answer"], "sources": result["sources"], "retries": 0,
            "steps": steps, "citations": []}


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
