"""Graph nodes: each one reads the State and returns ONLY the fields it changed.

LangGraph merges the returned dict into the State. A small dict makes it
obvious what each step is responsible for.

Flow (wired in graph.py):
    route -> (off_topic? generate : retrieve)
    retrieve -> grade -> (sufficient? generate : rewrite -> retrieve ...)
"""

import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app import router
from app.ingest import TOP_K, get_vectorstore
from app.state import State

# 0.8s vs 12.4s for gemini-3.5-flash on the same prompt, near-identical answer.
LLM_MODEL = "gemini-3.5-flash-lite"

# The retry cap. Without it, a question the papers can't answer loops
# forever, burning quota and never replying. 2 rewrites = at most 3 searches
# and 6 LLM calls: measured ~10s worst case, the edge of the 10s budget.
MAX_REWRITES = 2

# Comparison questions need evidence from two papers; with only 4 chunks,
# one paper usually takes all the slots. 8 = room for ~4 per paper.
COMPARISON_K = 2 * TOP_K

IDK = "I don't know based on the provided documents."


# ---------------------------------------------------------------- prompts ---

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about research papers using ONLY the numbered "
     "context blocks below. Cite the blocks you used like [1] or [2][3]. "
     "An answer may need to combine several blocks, e.g. a comparison where "
     "each method is described in a different block. "
     f"If the blocks contain no information that answers the question, reply "
     f"exactly: \"{IDK}\" Do not use outside knowledge."),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance grader for a question-answering system "
     "over research papers.\n"
     "1. For EACH numbered chunk, decide if it contains information that "
     "directly helps answer the question, or ANY PART of it: for a "
     "comparison, a chunk covering just one side is relevant. Being about "
     "the same paper or topic is NOT enough: the chunk must contain facts "
     "the answer would use.\n"
     "2. Decide if the relevant chunks TOGETHER are sufficient to answer "
     "the question fully and specifically. If a detail the question asks "
     "for (a mechanism, condition, number, name) is absent, it is NOT "
     "sufficient; say in `missing` exactly what information is still needed."),
    ("human", "Question: {question}\n\nChunks:\n\n{context}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write search queries for a semantic (vector) search over research "
     "papers about retrieval-augmented generation and LLM agents. "
     "Vector search matches meaning, so phrase the query the way the paper "
     "text itself would state the answer, using specific technical terms. "
     "Return ONLY the new query, nothing else."),
    ("human",
     "Original question: {question}\n"
     "Previous search query: {search_query}\n"
     "Information still missing: {missing}\n\n"
     "Write one new search query that targets the missing information."),
])


# ------------------------------------------------------- structured output ---

class ChunkGrade(BaseModel):
    chunk: int = Field(description="The chunk number, e.g. 2 for [2]")
    # The reason comes BEFORE the verdict on purpose: the model writes its
    # justification first, then decides, which makes the yes/no more careful.
    reason: str = Field(description="One short sentence: the useful fact it "
                                    "contains, or why it does not help")
    relevant: bool


class Grading(BaseModel):
    grades: list[ChunkGrade]
    sufficient: bool = Field(description="True only if the relevant chunks "
                                         "together answer the question fully")
    missing: str = Field(description="If not sufficient: the specific "
                                     "information still needed. Else empty.")


# ----------------------------------------------------------------- clients ---

# Opening Chroma and the API clients takes time, so do it once per process.
@lru_cache(maxsize=1)
def _vectorstore():
    return get_vectorstore()


# ---- timeouts and the fallback model (found while deploying, step 7) ----
# 1. Every network call needs a time limit: a Gemini call that never
#    answered once left a web request hanging for 5+ minutes.
# 2. At peak hours (Indian evening) the free tier slowed down PER MODEL:
#    gemini-3.5-flash-lite took 24-43 s for "reply OK" or returned 504,
#    while gemini-3.1-flash-lite answered in ~5 s. So each call tries the
#    main model briefly, then falls back to a different model, which also
#    has its own separate free quota.
#
# Off-peak, our slowest main-model call (CAG, two whole papers) takes ~5 s,
# so 20 s means "overloaded": stop waiting and use the fallback.
PRIMARY_TIMEOUT_S = 20
# The fallback is the last resort, so it gets more patience and one retry
# (worst case ~2 min per call instead of forever).
LLM_TIMEOUT_S = 60
LLM_MAX_RETRIES = 1
# Older lite model, a different deployment from the main one. The benchmark
# measured LLM_MODEL only; fallback answers may differ slightly in quality.
FALLBACK_MODEL = "gemini-3.1-flash-lite"


def _primary():
    # No retries on the main model: when it's overloaded, retrying it just
    # adds another 20 s before the fallback gets a chance.
    return ChatGoogleGenerativeAI(model=LLM_MODEL, timeout=PRIMARY_TIMEOUT_S, max_retries=0)


def _fallback():
    return ChatGoogleGenerativeAI(model=FALLBACK_MODEL, timeout=LLM_TIMEOUT_S,
                                  max_retries=LLM_MAX_RETRIES)


@lru_cache(maxsize=1)
def llm():
    """The chat model used everywhere: main model, fallback on any error."""
    return _primary().with_fallbacks([_fallback()])


def structured(schema):
    """Same main -> fallback chain, but the reply is forced into `schema`.

    with_structured_output forces the reply into a pydantic schema (JSON
    validated), so code reads real booleans instead of parsing free text
    like "Yes, mostly...". It has to be applied to each model before
    chaining the fallback, so it gets its own helper.
    """
    return _primary().with_structured_output(schema).with_fallbacks(
        [_fallback().with_structured_output(schema)])


@lru_cache(maxsize=1)
def _grader():
    # prompt | model: fill the prompt, then call the model with it.
    return GRADE_PROMPT | structured(Grading)


# ----------------------------------------------------------------- helpers ---

def _label(doc: Document) -> str:
    return f"{doc.metadata['source']} p.{doc.metadata['page']}"


def _format_context(docs: list[Document]) -> str:
    """Number each chunk so the model can cite/grade it and we can map [n] back."""
    return "\n\n".join(f"[{i}] ({_label(d)})\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


def _cited_sources(answer: str, docs: list[Document]) -> list[str]:
    """Only the chunks the answer actually cites, e.g. [2] -> docs[1]."""
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    labels = [_label(docs[n - 1]) for n in cited if 1 <= n <= len(docs)]
    # dict.fromkeys removes duplicates but keeps order (two chunks, same page)
    return list(dict.fromkeys(labels))


# ------------------------------------------------------------------- nodes ---

def route(state: State) -> dict:
    """Classify the question locally (~1 ms, no API call)."""
    return {"route": router.predict(state["question"])}


def retrieve(state: State) -> dict:
    """Search with the current query, skipping chunks already graded.

    Without the skip, summary-like chunks (abstract, intro, conclusion)
    win every search, and each retry re-grades the same 4 chunks. Tested:
    the CRAG "Action Trigger" chunk ranked #8 and was never reached.

    Chunks judged relevant earlier stay in `documents`, so a retry can only
    add evidence, never lose it.
    """
    query = state.get("search_query") or state["question"]
    k = COMPARISON_K if state.get("route") == "comparison" else TOP_K
    seen = set(state.get("seen_ids", []))
    # Ask for extra results so k remain after removing seen ones.
    hits = _vectorstore().similarity_search(query, k=k + len(seen))
    new = [d for d in hits if d.id not in seen][:k]
    return {
        "documents": state.get("documents", []) + new,
        "seen_ids": [*seen, *(d.id for d in new)],
        "search_query": query,
    }


def grade(state: State) -> dict:
    """One LLM call grades all chunks at once and judges sufficiency.

    Grading against the ORIGINAL question, not the rewritten query: the
    rewrite is only a search tool; the user's question is what must be
    answered.
    """
    docs = state["documents"]
    result: Grading = _grader().invoke({
        "question": state["question"],
        "context": _format_context(docs),
    })
    relevant_ids = {g.chunk for g in result.grades if g.relevant}
    relevant = [d for i, d in enumerate(docs, start=1) if i in relevant_ids]
    return {
        "documents": relevant,
        # No relevant chunks can never be sufficient, whatever the model says.
        "sufficient": result.sufficient and bool(relevant),
        "missing": result.missing,
    }


def rewrite(state: State) -> dict:
    """Write a better search query aimed at what the grader said is missing."""
    messages = REWRITE_PROMPT.invoke({
        "question": state["question"],
        "search_query": state.get("search_query") or state["question"],
        "missing": state.get("missing") or "relevant information",
    })
    new_query = llm().invoke(messages).text.strip().strip('"')
    return {"search_query": new_query,
            "attempts": state.get("attempts", 0) + 1}


def generate(state: State) -> dict:
    docs = state.get("documents", [])
    if not docs:
        # Nothing relevant survived grading: don't pay for an LLM call
        # that can only say "I don't know".
        return {"answer": IDK, "sources": []}
    messages = ANSWER_PROMPT.invoke({"context": _format_context(docs),
                                     "question": state["question"]})
    answer = llm().invoke(messages).text
    return {"answer": answer, "sources": _cited_sources(answer, docs)}


# ------------------------------------------------------ conditional edges ---

def decide_after_route(state: State) -> str:
    """Off-topic: skip search entirely. generate() sees no documents and
    returns "I don't know" without any API call."""
    return "generate" if state.get("route") == "off_topic" else "retrieve"


def decide_after_grade(state: State) -> str:
    """The branch: answer now, or rewrite and search again?

    Returns the NAME of the next node. Reaching the cap with insufficient
    evidence still goes to generate, which answers from whatever relevant
    chunks exist or says "I don't know". Never loop forever.
    """
    if state.get("sufficient"):
        return "generate"
    if state.get("attempts", 0) >= MAX_REWRITES:
        return "generate"
    return "rewrite"
