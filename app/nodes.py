"""Graph nodes: each one reads the State and returns ONLY the fields it changed.

LangGraph merges the returned dict into the State. A small dict makes it
obvious what each step is responsible for.

Flow (wired in graph.py):
    retrieve -> grade -> (sufficient? generate : rewrite -> retrieve ...)
"""

import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.ingest import TOP_K, get_vectorstore
from app.state import State

# 0.8s vs 12.4s for gemini-3.5-flash on the same prompt, near-identical answer.
LLM_MODEL = "gemini-3.5-flash-lite"

# The retry cap. Without it, a question the papers can't answer loops
# forever, burning quota and never replying. 2 rewrites = at most 3 searches:
# worst case 6 LLM calls (~5s), inside the 10s budget.
MAX_REWRITES = 2

IDK = "I don't know based on the provided documents."


# ---------------------------------------------------------------- prompts ---

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about research papers using ONLY the numbered "
     "context blocks below. Cite the blocks you used like [1] or [2][3]. "
     f"If the context does not contain the answer, reply exactly: \"{IDK}\" "
     "Do not use outside knowledge."),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance grader for a question-answering system "
     "over research papers.\n"
     "1. For EACH numbered chunk, decide if it contains information that "
     "directly helps answer the question. Being about the same paper or "
     "topic is NOT enough: the chunk must contain facts the answer would use.\n"
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


@lru_cache(maxsize=1)
def llm():
    # max_retries: the free tier answers some calls with 429/503; the client
    # backs off and retries instead of failing the whole question.
    return ChatGoogleGenerativeAI(model=LLM_MODEL, max_retries=3)


@lru_cache(maxsize=1)
def _grader():
    # prompt | model: fill the prompt, then call the model with it.
    # with_structured_output forces the reply into the Grading schema (JSON
    # validated by pydantic), so the code reads real booleans instead of
    # parsing free text like "Yes, mostly...".
    return GRADE_PROMPT | llm().with_structured_output(Grading)


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

def retrieve(state: State) -> dict:
    """Search with the current query, skipping chunks already graded.

    Without the skip, summary-like chunks (abstract, intro, conclusion)
    win every search, and each retry re-grades the same 4 chunks. Tested:
    the CRAG "Action Trigger" chunk ranked #8 and was never reached.

    Chunks judged relevant earlier stay in `documents`, so a retry can only
    add evidence, never lose it.
    """
    query = state.get("search_query") or state["question"]
    seen = set(state.get("seen_ids", []))
    # Ask for extra results so TOP_K remain after removing seen ones.
    hits = _vectorstore().similarity_search(query, k=TOP_K + len(seen))
    new = [d for d in hits if d.id not in seen][:TOP_K]
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


# ------------------------------------------------------- conditional edge ---

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
