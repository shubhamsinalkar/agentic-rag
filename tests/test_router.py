"""Router: no API calls needed (the model runs locally from models/router.pkl)."""

import json

import numpy as np
import pytest

from app import router
from app.graph import app
from app.ingest import ROOT
from app.nodes import IDK, decide_after_route

BENCH = json.loads((ROOT / "data" / "testset.json").read_text(encoding="utf-8"))


def test_never_refuses_an_answerable_benchmark_question():
    """The expensive mistake: refusing a question the papers can answer."""
    refused = [q["id"] for q in BENCH
               if q["type"] != "out_of_scope" and router.predict(q["question"]) == "off_topic"]
    assert refused == []


@pytest.mark.parametrize("question", [
    "Who won the 2022 FIFA World Cup?",       # benchmark q29, never trained on
    "How do I bake sourdough bread?",
    "What is the best laptop for gaming?",
])
def test_catches_obvious_off_topic(question):
    assert router.predict(question) == "off_topic"


def test_recognises_a_comparison():
    assert router.predict("Compare how Self-RAG and Adaptive-RAG decide to retrieve.") == "comparison"


@pytest.mark.parametrize("proba, expected", [
    ([0.1, 0.1, 0.1, 0.7], "off_topic"),   # confident: refuse
    ([0.1, 0.2, 0.35, 0.35], "factual"),     # tied with factual: never refuse
    ([0.1, 0.3, 0.2, 0.39], "conceptual"),   # off_topic leads but < 0.4: runner-up
])
def test_choose_route_only_refuses_when_confident(proba, expected):
    classes = ["comparison", "conceptual", "factual", "off_topic"]
    assert router.choose_route(np.array(proba), classes, 0.4) == expected


def test_off_topic_skips_search_entirely():
    """route -> generate directly: no retrieval, no grading, no LLM call."""
    final = app.invoke({"question": "How do I bake sourdough bread?"})
    assert final["route"] == "off_topic"
    assert final["answer"] == IDK
    assert "seen_ids" not in final  # retrieve never ran


@pytest.mark.parametrize("route, expected", [
    ("off_topic", "generate"), ("factual", "retrieve"),
    ("conceptual", "retrieve"), ("comparison", "retrieve")])
def test_decide_after_route(route, expected):
    assert decide_after_route({"route": route}) == expected


def test_saved_model_matches_installed_sklearn():
    import sklearn
    assert router._bundle()["sklearn_version"] == sklearn.__version__
