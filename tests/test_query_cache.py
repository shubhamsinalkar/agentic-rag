"""The query cache: no API needed (a fake embedding model counts its calls)."""

from langchain_core.embeddings import Embeddings

from app.query_cache import CachedQueryEmbeddings


class CountingFake(Embeddings):
    def __init__(self):
        self.query_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return [float(len(text)), 1.0]

    def embed_documents(self, texts):
        return [[0.0, 0.0] for _ in texts]


def test_same_question_is_embedded_once(tmp_path):
    fake = CountingFake()
    cache = CachedQueryEmbeddings(fake, tmp_path / "c.sqlite", "model-a@768")
    first = cache.embed_query("What is CRAG?")
    second = cache.embed_query("What is CRAG?")
    assert first == second == [13.0, 1.0]
    assert fake.query_calls == 1


def test_cache_survives_a_restart(tmp_path):
    path = tmp_path / "c.sqlite"
    CachedQueryEmbeddings(CountingFake(), path, "model-a@768").embed_query("q")
    fresh = CountingFake()
    CachedQueryEmbeddings(fresh, path, "model-a@768").embed_query("q")
    assert fresh.query_calls == 0  # served from disk


def test_a_different_model_never_reuses_old_vectors(tmp_path):
    path = tmp_path / "c.sqlite"
    CachedQueryEmbeddings(CountingFake(), path, "model-a@768").embed_query("q")
    other = CountingFake()
    CachedQueryEmbeddings(other, path, "model-b@768").embed_query("q")
    assert other.query_calls == 1
