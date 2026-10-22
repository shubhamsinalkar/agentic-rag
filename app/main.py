"""FastAPI service: POST /ask, GET /health, and a small demo page at /.

Run locally:  .venv\\Scripts\\uvicorn app.main:app --reload
Then open:    http://127.0.0.1:8000
"""

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from google.genai.errors import ClientError
from pydantic import BaseModel, Field

from app import nodes, router
from app.ingest import DOCS_DIR
from app.strategies import STRATEGIES

log = logging.getLogger("agentic_rag")
STATIC = Path(__file__).parent / "static"

# The public demo runs on a free API quota (about 1,000 embedding and LLM
# requests per day). Without limits, one visitor's script could use it up
# in minutes. An agentic question costs ~3 LLM calls, so 150 questions/day
# stays inside the quota with headroom. Both are env-configurable.
PER_IP_PER_MINUTE = int(os.getenv("PER_IP_PER_MINUTE", "5"))
QUESTIONS_PER_DAY = int(os.getenv("QUESTIONS_PER_DAY", "150"))


class RateLimiter:
    """In-memory limits: per-IP sliding minute + one global daily budget.

    In-memory is enough for a single small server; it resets on restart.
    Several servers would need a shared store such as Redis.
    """

    def __init__(self):
        self._lock = Lock()  # requests run in a thread pool
        self._recent: dict[str, deque] = defaultdict(deque)
        self._day, self._count = date.today(), 0

    def check(self, ip: str) -> None:
        now = time.monotonic()
        with self._lock:
            if date.today() != self._day:
                self._day, self._count = date.today(), 0
            if self._count >= QUESTIONS_PER_DAY:
                raise HTTPException(503, "The demo's daily question budget is used up. "
                                         "Please try again tomorrow.")
            recent = self._recent[ip]
            while recent and now - recent[0] > 60:
                recent.popleft()
            if len(recent) >= PER_IP_PER_MINUTE:
                raise HTTPException(429, "Too many questions. Please wait a minute.")
            recent.append(now)
            self._count += 1


limiter = RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the index, router and API clients once at start-up, so the first
    # visitor doesn't pay the 5-10 s warm-up cost.
    store = nodes._vectorstore()
    router._bundle()
    nodes.llm()
    log.warning("ready: %d chunks indexed", store._collection.count())
    yield


app = FastAPI(title="Agentic RAG", lifespan=lifespan,
              description="Question answering over 6 RAG research papers, with "
                          "self-correcting retrieval.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    strategy: Literal["agentic_rag", "plain_rag", "cag"] = "agentic_rag"


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    strategy: str
    route: str | None = None   # router's label (agentic strategy only)
    retries: int
    latency_s: float


def _client_ip(request: Request) -> str:
    # Behind Render's proxy the real client is the first X-Forwarded-For
    # entry. A client could fake it, which only matters for the per-IP limit;
    # the daily budget still caps the total.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# A plain `def` (not async): the graph makes blocking network calls, and
# FastAPI runs sync endpoints in a thread pool so they don't block others.
@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    limiter.check(_client_ip(request))
    started = time.perf_counter()
    try:
        out = STRATEGIES[body.strategy](body.question.strip())
    except Exception as err:
        cause = err.__cause__ or err
        if isinstance(cause, ClientError) and cause.code == 429:
            raise HTTPException(503, "The free AI quota is used up for now. "
                                     "Please try again later.") from err
        log.exception("ask failed")
        raise HTTPException(500, "Something went wrong answering that question.") from err
    return AskResponse(answer=out["answer"], sources=out["sources"], strategy=body.strategy,
                       route=out.get("route"), retries=out["retries"],
                       latency_s=round(time.perf_counter() - started, 2))


@app.get("/health")
def health() -> dict:
    return {"status": "ok",
            "chunks": nodes._vectorstore()._collection.count(),
            "papers": len(list(DOCS_DIR.glob("*.pdf")))}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
