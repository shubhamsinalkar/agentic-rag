"""Remember question embeddings on disk so the same question is embedded once.

Why: the free tier allows 1,000 embedding requests per day, and every
search embeds its query. Tests and benchmark reruns ask the same questions
again and again; without a cache they burned the daily quota.

Only QUERY embeddings are cached. Document embeddings live in Chroma
already, and are only computed when the index is rebuilt.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

from langchain_core.embeddings import Embeddings


class CachedQueryEmbeddings(Embeddings):
    """Wraps an embedding model; looks up query vectors before calling it.

    The cache key includes the model name and size: vectors from a different
    model live in a different "space", and silently mixing them would ruin
    search with no error (the trap from step 1).
    """

    def __init__(self, inner: Embeddings, path: Path, model_id: str):
        self.inner = inner
        self.path = path
        self.model_id = model_id
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector TEXT)")

    def _connect(self) -> sqlite3.Connection:
        # A new connection per call: safe when the web server answers
        # several requests at once (one sqlite connection can't be shared
        # across threads).
        return sqlite3.connect(self.path)

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_id}\n{text}".encode()).hexdigest()

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        with self._connect() as db:
            row = db.execute("SELECT vector FROM vectors WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        vector = self.inner.embed_query(text)
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO vectors VALUES (?, ?)", (key, json.dumps(vector)))
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)
