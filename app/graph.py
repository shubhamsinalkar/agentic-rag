"""The LangGraph wiring: routed, self-correcting retrieval.

    START -> route --off_topic-----------------------------> generate -> END
               |                                               ^
               +--> retrieve -> grade --sufficient-------------+
                       ^          |
                       |          +--not sufficient (attempts < 2)--> rewrite
                       +-----------------------------------------------+

Ask a question:  .venv\\Scripts\\python -m app.graph "your question"
(prints every step, so you can watch the loop fire)
"""

import sys
import time

from langgraph.graph import END, START, StateGraph

from app.nodes import (decide_after_grade, decide_after_route, generate, grade,
                       retrieve, rewrite, route)
from app.state import State


def build_graph():
    graph = StateGraph(State)

    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", decide_after_route,
                                ["retrieve", "generate"])
    graph.add_edge("retrieve", "grade")
    # The conditional edge: after grade, decide_after_grade(state) returns
    # the name of the next node. The list declares every possible target.
    graph.add_conditional_edges("grade", decide_after_grade,
                                ["generate", "rewrite"])
    graph.add_edge("rewrite", "retrieve")  # this edge closes the loop
    graph.add_edge("generate", END)

    return graph.compile()


app = build_graph()

# LangGraph's own backstop: it raises an error after this many steps. Our
# MAX_REWRITES cap stops far earlier (at most 9 steps); this only catches a
# bug in our own cap.
RECURSION_LIMIT = 15


def ask(question: str) -> dict:
    """Run the full graph and return the final state."""
    return app.invoke({"question": question},
                      config={"recursion_limit": RECURSION_LIMIT})


def _show(node: str, update: dict) -> None:
    """One line per step, so the path through the graph is visible."""
    if node == "route":
        print(f"route     {update['route']}")
    elif node == "retrieve":
        print(f"retrieve  query={update['search_query']!r}\n"
              f"          -> {len(update['documents'])} chunks to grade "
              f"({len(update['seen_ids'])} seen so far)")
    elif node == "grade":
        verdict = "SUFFICIENT" if update["sufficient"] else "NOT sufficient"
        print(f"grade     {len(update['documents'])} relevant, {verdict}")
        if not update["sufficient"]:
            print(f"          missing: {update['missing']}")
    elif node == "rewrite":
        print(f"rewrite   attempt {update['attempts']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m app.graph "your question"')
    question = " ".join(sys.argv[1:])
    started = time.perf_counter()
    final = {}
    # stream_mode="updates" yields {node_name: what_that_node_returned}
    for step in app.stream({"question": question}, stream_mode="updates",
                           config={"recursion_limit": RECURSION_LIMIT}):
        for node, update in step.items():
            _show(node, update)
            final.update(update)
    print(f"\n{final['answer']}")
    print("\nSources:", ", ".join(final["sources"]) or "none")
    print(f"({time.perf_counter() - started:.1f}s)")
