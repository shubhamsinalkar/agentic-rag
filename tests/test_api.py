"""The web API. Off-topic questions skip every LLM call, so most of these
tests run without spending quota; failures are simulated."""

import pytest
from fastapi.testclient import TestClient
from google.genai.errors import ClientError

from app import main
from app.nodes import IDK

OFF_TOPIC = "How do I bake sourdough bread?"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "limiter", main.RateLimiter())  # fresh limits per test
    with TestClient(main.app) as c:  # `with` runs the start-up warm-up
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 635
    assert body["papers"] == 6


def test_serves_the_website_and_its_assets(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>Agentic RAG" in r.text
    # the built JS bundle referenced by index.html must be served too
    js = r.text.split('src="')[1].split('"')[0]
    assert js.startswith("/assets/") and client.get(js).status_code == 200
    assert client.get("/favicon.svg").status_code == 200


def test_api_routes_win_over_the_website(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/ask", json={}).status_code == 422  # API, not a static 404/405


@pytest.mark.parametrize("payload", [
    {"question": "hi"},                                  # too short
    {"question": "x" * 501},                             # too long
    {"question": "What is CRAG?", "strategy": "magic"},  # unknown strategy
    {},                                                  # missing
])
def test_rejects_bad_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_off_topic_question_end_to_end(client):
    body = client.post("/ask", json={"question": OFF_TOPIC}).json()
    assert body["answer"] == IDK
    assert body["route"] == "off_topic"
    assert body["sources"] == [] and body["retries"] == 0
    # The trace shows the shortcut: classified, then answered, no search.
    assert [s["step"] for s in body["steps"]] == ["route", "answer"]
    assert body["steps"][0]["label"] == "off_topic"
    assert body["citations"] == []


def test_per_ip_rate_limit(client, monkeypatch):
    monkeypatch.setattr(main, "PER_IP_PER_MINUTE", 2)
    codes = [client.post("/ask", json={"question": OFF_TOPIC}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_daily_budget(client, monkeypatch):
    monkeypatch.setattr(main, "QUESTIONS_PER_DAY", 1)
    assert client.post("/ask", json={"question": OFF_TOPIC}).status_code == 200
    r = client.post("/ask", json={"question": OFF_TOPIC}, headers={"x-forwarded-for": "9.9.9.9"})
    assert r.status_code == 503 and "budget" in r.json()["detail"]


def test_quota_errors_become_a_friendly_503(client, monkeypatch):
    def out_of_quota(question):
        try:
            raise ClientError(429, {"error": {"message": "RESOURCE_EXHAUSTED"}})
        except ClientError as e:
            raise RuntimeError("wrapped, like LangChain does") from e
    monkeypatch.setitem(main.STRATEGIES, "plain_rag", out_of_quota)
    r = client.post("/ask", json={"question": "What is CRAG?", "strategy": "plain_rag"})
    assert r.status_code == 503 and "quota" in r.json()["detail"]


def test_google_overload_becomes_a_busy_503(client, monkeypatch):
    """Seen live at peak hours: 504 DEADLINE_EXCEEDED from the free tier."""
    from langchain_google_genai.chat_models import GoogleAPIError

    def overloaded(question):
        raise GoogleAPIError(504, {"error": {"message": "Deadline expired"}})
    monkeypatch.setitem(main.STRATEGIES, "plain_rag", overloaded)
    r = client.post("/ask", json={"question": "What is CRAG?", "strategy": "plain_rag"})
    assert r.status_code == 503 and "overloaded" in r.json()["detail"]


def test_unexpected_errors_hide_details(client, monkeypatch):
    def boom(question):
        raise ValueError("secret internal detail")
    monkeypatch.setitem(main.STRATEGIES, "plain_rag", boom)
    r = client.post("/ask", json={"question": "What is CRAG?", "strategy": "plain_rag"})
    assert r.status_code == 500 and "secret" not in r.text
