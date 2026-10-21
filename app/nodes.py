"""Graph nodes: each one reads the State and returns ONLY the fields it changed.

LangGraph merges the returned dict into the State. Returning the whole State
would also work, but a small dict makes it obvious what each step is
responsible for.
"""

import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from app.ingest import TOP_K, get_vectorstore
from app.state import State

LLM_MODEL = "gemini-3.5-flash-lite"

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about research papers using ONLY the numbered "
     "context blocks below. Cite the blocks you used like [1] or [2][3]. "
     "If the context does not contain the answer, reply exactly: "
     "\"I don't know based on the provided documents.\" "
     "Do not use outside knowledge."),
    ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
])


# Opening Chroma and the API clients takes time, so do it once per process
# and reuse them for every question.
@lru_cache(maxsize=1)
def _vectorstore():
    return get_vectorstore()


@lru_cache(maxsize=1)
def _llm():
    # max_retries: the free tier answers some calls with 429/503; the client
    # backs off and retries instead of failing the whole question.
    return ChatGoogleGenerativeAI(model=LLM_MODEL, max_retries=3)


def _label(doc: Document) -> str:
    return f"{doc.metadata['source']} p.{doc.metadata['page']}"


def _format_context(docs: list[Document]) -> str:
    """Number each chunk so the model can cite it and we can map [n] back."""
    return "\n\n".join(f"[{i}] ({_label(d)})\n{d.page_content}"
                       for i, d in enumerate(docs, start=1))


# ------------------------------------------------------------------ nodes ---

def retrieve(state: State) -> dict:
    docs = _vectorstore().similarity_search(state["question"], k=TOP_K)
    return {"documents": docs}


def generate(state: State) -> dict:
    docs = state["documents"]
    messages = ANSWER_PROMPT.invoke({"context": _format_context(docs),
                                     "question": state["question"]})
    answer = _llm().invoke(messages).text
    return {"answer": answer, "sources": _cited_sources(answer, docs)}


def _cited_sources(answer: str, docs: list[Document]) -> list[str]:
    """Only the chunks the answer actually cites, e.g. [2] -> docs[1].

    Listing every retrieved chunk would show sources even for
    "I don't know", which misleads the reader.
    """
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    labels = [_label(docs[n - 1]) for n in cited if 1 <= n <= len(docs)]
    # dict.fromkeys removes duplicates but keeps order (two chunks, same page)
    return list(dict.fromkeys(labels))
