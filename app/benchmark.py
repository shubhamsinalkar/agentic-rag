"""Step 6: run every test question through every strategy, then grade answers.

    .venv\\Scripts\\python -m app.benchmark run     # ~30 min on the free tier
    .venv\\Scripts\\python -m app.benchmark judge   # LLM-as-judge, ~15 min

Both steps append one JSON line per result and skip work already done, so
a crash (Wi-Fi, rate limit) loses nothing: just run the command again.
Analysis and charts: notebooks/02_benchmark.ipynb
"""

import json
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers.context import register_configure_hook
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.ingest import ROOT
from app.nodes import IDK
from app.strategies import STRATEGIES

OUT_DIR = ROOT / "data" / "benchmark"
RUNS = OUT_DIR / "runs.jsonl"
JUDGMENTS = OUT_DIR / "judgments.jsonl"
TESTSET = ROOT / "data" / "testset.json"

# Free tier allows roughly 10-15 LLM requests/min. One question across the
# three strategies makes ~6 calls; a 6s pause between runs keeps us near
# 10/min, so rate-limit back-offs don't inflate the measured latency.
PAUSE_S = 6

# A different model from the one that wrote the answers: a model grading its
# own output tends to be lenient with it (self-preference bias).
# The first choice, gemini-3.5-flash, allows only 20 free requests per day,
# so it graded just 20 answers (kept in judgments_flash_sample.jsonl as a
# cross-check). gemini-3.1-flash-lite is an older, separate model with a
# much larger free quota; every verdict is also audited by hand.
JUDGE_MODEL = "gemini-3.1-flash-lite"


# ------------------------------------------------------------ measurement ---

class Meter(BaseCallbackHandler):
    """Counts LLM calls and tokens for everything run inside `metered()`.

    Embedding calls are not LLM calls and aren't counted; they are a few
    hundred tokens per search, negligible next to the prompts.
    """

    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = self.cached_tokens = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        usage = getattr(response.generations[0][0].message, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.cached_tokens += usage.get("input_token_details", {}).get("cache_read", 0) or 0


# Same mechanism as langchain_core's get_usage_metadata_callback: a context
# variable that LangChain checks on every model call, so calls made deep
# inside the graph are counted without passing the meter around.
_meter_var: ContextVar[Meter | None] = ContextVar("benchmark_meter", default=None)
register_configure_hook(_meter_var, inheritable=True)


@contextmanager
def metered(meter: Meter):
    token = _meter_var.set(meter)
    try:
        yield meter
    finally:
        _meter_var.reset(token)


# ------------------------------------------------------------------ files ---

def _read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append(path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _testset() -> list[dict]:
    return json.loads(TESTSET.read_text(encoding="utf-8"))


# -------------------------------------------------------------------- run ---

def run() -> None:
    done = {(r["id"], r["strategy"]) for r in _read(RUNS)}
    todo = [(q, name) for q in _testset() for name in STRATEGIES
            if (q["id"], name) not in done]
    print(f"{len(done)} runs already saved, {len(todo)} to go")
    # Strategies interleave per question (plain, agentic, cag, next question...)
    # so a slow minute on the free tier hits all three equally, not just one.
    for i, (q, name) in enumerate(todo, start=1):
        meter = Meter()
        started = time.perf_counter()
        try:
            with metered(meter):
                out = STRATEGIES[name](q["question"])
        except Exception as err:  # keep going; a rerun retries the gap
            print(f"[{i}/{len(todo)}] {q['id']} {name:12} FAILED: {str(err)[:120]}")
            time.sleep(PAUSE_S * 5)
            continue
        latency = time.perf_counter() - started
        _append(RUNS, {
            "id": q["id"], "strategy": name, "answer": out["answer"],
            "sources": out["sources"], "retries": out["retries"],
            "latency_s": round(latency, 2), "llm_calls": meter.calls,
            "input_tokens": meter.input_tokens, "output_tokens": meter.output_tokens,
            "cached_tokens": meter.cached_tokens,
        })
        print(f"[{i}/{len(todo)}] {q['id']} {name:12} {latency:5.1f}s "
              f"calls={meter.calls} retries={out['retries']}")
        time.sleep(PAUSE_S)


# ------------------------------------------------------------------ judge ---

class Verdict(BaseModel):
    # Reason first: the judge explains, then decides (same trick as the grader).
    reason: str = Field(description="One or two sentences comparing the "
                                    "candidate with the reference")
    verdict: Literal["correct", "partial", "incorrect"]


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade answers from a question-answering system against a reference "
     "answer written by a human.\n"
     "- correct: contains all the key facts of the reference and contradicts "
     "none. Extra correct detail, different wording and citations are fine.\n"
     "- partial: some key facts right, some missing, nothing contradicted.\n"
     "- incorrect: wrong, contradicts the reference, or fails to answer.\n"
     "If the reference says NOT IN DOCUMENTS, the candidate is correct only "
     "if it declines to answer (e.g. says it doesn't know); any attempted "
     "answer is incorrect."),
    ("human", "Question: {question}\n\nReference answer: {reference}\n\n"
              "Candidate answer: {answer}"),
])


def _is_bare_idk(answer: str) -> bool:
    """The exact refusal sentence and nothing else (no partial answer)."""
    return answer.strip().rstrip(".").lower() == IDK.rstrip(".").lower()


def judge() -> None:
    questions = {q["id"]: q for q in _testset()}
    done = {(j["id"], j["strategy"]) for j in _read(JUDGMENTS)}
    todo = [r for r in _read(RUNS) if (r["id"], r["strategy"]) not in done]
    print(f"{len(done)} already judged, {len(todo)} to go")
    grader = JUDGE_PROMPT | ChatGoogleGenerativeAI(
        model=JUDGE_MODEL, max_retries=3).with_structured_output(Verdict)

    for i, r in enumerate(todo, start=1):
        q = questions[r["id"]]
        if _is_bare_idk(r["answer"]):
            # No LLM needed: a bare refusal is right exactly when the
            # question is out of scope.
            ok = q["type"] == "out_of_scope"
            record = {"verdict": "correct" if ok else "incorrect", "judge": "rule",
                      "reason": "bare 'I don't know'"}
        else:
            try:
                v: Verdict = grader.invoke({"question": q["question"],
                                            "reference": q["expected_answer"],
                                            "answer": r["answer"]})
            except Exception as err:
                print(f"[{i}/{len(todo)}] {r['id']} {r['strategy']:12} FAILED: {str(err)[:120]}")
                time.sleep(PAUSE_S * 5)
                continue
            record = {"verdict": v.verdict, "judge": JUDGE_MODEL, "reason": v.reason}
            time.sleep(PAUSE_S)
        _append(JUDGMENTS, {"id": r["id"], "strategy": r["strategy"], **record})
        print(f"[{i}/{len(todo)}] {r['id']} {r['strategy']:12} {record['verdict']}")


if __name__ == "__main__":
    commands = {"run": run, "judge": judge}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("usage: python -m app.benchmark run|judge")
    commands[sys.argv[1]]()
