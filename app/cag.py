"""CAG (cache-augmented generation): no search, just put whole papers in the prompt.

Why per-paper instead of the whole corpus:
  All 6 papers = 142k tokens. That fits Gemini's 1M window, but on the free
  tier each question took 2-4 minutes (tokens-per-minute throttling).
  One paper (~15-28k tokens) answers in ~3-4s. So: a cheap LLM call picks
  the 1-2 papers most likely to hold the answer, then those papers go in
  whole.

Where the "cache" comes in:
  Gemini caches repeated prompt prefixes automatically (implicit caching).
  The papers go FIRST in the prompt and the question LAST, so two questions
  about the same paper share an identical prefix: measured 85% of input
  tokens served from cache on the second question.

Try it:  .venv\\Scripts\\python -m app.cag "your question"
"""

import re
import sys
from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.ingest import DOCS_DIR, PdfReader
from app.nodes import IDK, llm, structured

# Frozen corpus, so a hand-written catalog is fine. The selector reads only
# these lines, never the papers, so the choice costs ~300 tokens.
PAPERS = {
    "chexnet.pdf": "CheXNet (medicine): a 121-layer DenseNet that detects "
                   "pneumonia and 13 other diseases on chest X-rays "
                   "(ChestX-ray14), compared against four radiologists.",
    "four_degrees.pdf": "Four Degrees of Separation (social networks): average "
                        "distance between Facebook users (721M users), Milgram's "
                        "small-world experiment, HyperANF, spid.",
    "generative_ai_at_work.pdf": "Generative AI at Work (economics): a GPT-based "
                                 "assistant for 5,000+ customer support agents; "
                                 "productivity, worker skill, learning, attrition.",
    "gravitational_waves.pdf": "Observation of Gravitational Waves (physics): "
                               "LIGO's first detection, GW150914, a binary black "
                               "hole merger; detectors, masses, significance.",
    "planet_nine.pdf": "Evidence for a Distant Giant Planet (astronomy): clustered "
                       "orbits of distant Kuiper belt objects explained by a "
                       "~10 Earth-mass planet (Planet Nine).",
    "transformer.pdf": "Attention Is All You Need (computer science): the "
                       "Transformer, multi-head self-attention, positional "
                       "encoding, machine translation BLEU results.",
}

# 2 papers ~ 30-55k tokens: enough for comparison questions, still fast.
MAX_PAPERS = 2


class PaperChoice(BaseModel):
    # Literal = the model can only pick real file names, never invent one.
    papers: list[Literal[tuple(PAPERS)]] = Field(
        description=f"The 1-{MAX_PAPERS} papers most likely to contain the "
                    "answer. Empty list if no paper is relevant.")


SELECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Pick which research papers are needed to answer the question.\n\n"
               "Papers:\n{catalog}"),
    ("human", "{question}"),
])

ANSWER_INSTRUCTIONS = (
    "You answer questions about the research papers below using ONLY their "
    "text. Each page starts with a header like === paper.pdf p.N ===. "
    "Cite the pages you used like [paper.pdf p.N]. "
    f"If the papers do not contain the answer, reply exactly: \"{IDK}\" "
    "Do not use outside knowledge."
)


@lru_cache(maxsize=None)
def paper_text(name: str) -> str:
    """Whole paper with a header per page, so the model can cite pages."""
    reader = PdfReader(DOCS_DIR / name)
    return "\n\n".join(f"=== {name} p.{i} ===\n{page.extract_text() or ''}"
                       for i, page in enumerate(reader.pages, start=1))


@lru_cache(maxsize=1)
def _selector():
    return SELECT_PROMPT | structured(PaperChoice)


def select_papers(question: str) -> list[str]:
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in PAPERS.items())
    choice: PaperChoice = _selector().invoke({"catalog": catalog,
                                              "question": question})
    # Sorted + de-duplicated: the same set of papers always produces the
    # same prompt prefix, which is what lets the cache hit.
    return sorted(set(choice.papers))[:MAX_PAPERS]


def answer(question: str) -> dict:
    return answer_from(question, select_papers(question))


def answer_from(question: str, papers: list[str]) -> dict:
    """Answer with the given papers in the prompt (split out so the API can
    time and report the two steps separately)."""
    if not papers:
        return {"answer": IDK, "sources": [], "papers": []}
    # Papers first (stable, cacheable prefix), question last (changes).
    system = ANSWER_INSTRUCTIONS + "\n\n" + "\n\n".join(paper_text(p) for p in papers)
    text = llm().invoke([SystemMessage(system), HumanMessage(question)]).text
    return {"answer": text, "sources": cited_pages(text, papers), "papers": papers}


def cited_pages(text: str, papers: list[str]) -> list[str]:
    """Parse citations like [a.pdf p.5], [a.pdf p. 5] and [a.pdf p.2, p.4].

    Only pages of papers we actually gave the model count; anything else
    would be a made-up citation.
    """
    sources = []
    for name, pages in re.findall(r"([\w-]+\.pdf)((?:[\s,]*p\.\s*\d+)+)", text):
        if name in papers:
            sources += [f"{name} p.{n}" for n in re.findall(r"\d+", pages)]
    return list(dict.fromkeys(sources))  # de-duplicate, keep order


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python -m app.cag "your question"')
    result = answer(" ".join(sys.argv[1:]))
    print("papers:", ", ".join(result["papers"]) or "none")
    print(f"\n{result['answer']}\n\nSources:", ", ".join(result["sources"]) or "none")
