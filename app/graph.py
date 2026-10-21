"""The LangGraph wiring. Step 2: START -> retrieve -> generate -> END.

Ask a question:  .venv\\Scripts\\python -m app.graph "your question"
"""

import sys
import time

from langgraph.graph import END, START, StateGraph

from app.nodes import generate, retrieve
from app.state import State


def build_graph():
    graph = StateGraph(State)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    # Step 3 replaces retrieve -> generate with retrieve -> grade and a
    # conditional edge: relevant -> generate, not relevant -> rewrite -> retrieve.

    return graph.compile()


app = build_graph()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m app.graph "your question"')
    question = " ".join(sys.argv[1:])
    started = time.perf_counter()
    result = app.invoke({"question": question})
    print(result["answer"])
    print("\nSources:", ", ".join(result["sources"]))
    print(f"({time.perf_counter() - started:.1f}s)")
