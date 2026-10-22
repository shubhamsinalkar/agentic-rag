"""The frozen corpus: six arXiv papers, pinned to exact versions.

The PDFs are NOT in the repository: three of them (RAG, CRAG, Lost in the
Middle) carry arXiv's standard license, which lets arXiv distribute them
but not us. Anyone (and the Render build) fetches them from arXiv instead:

    .venv\\Scripts\\python -m app.corpus

Each file is checked against a SHA-256 hash, so everyone gets byte-identical
PDFs. That matters: the shipped search index (data/index/) only works if the
PDFs split into exactly the same chunks.
"""

import hashlib
import sys
import time
import urllib.request
from dataclasses import dataclass

from app.ingest import DOCS_DIR


@dataclass(frozen=True)
class Paper:
    file: str
    arxiv: str      # id + version: a new version would change the text
    sha256: str
    license: str


CORPUS = [
    Paper("adaptive_rag.pdf", "2403.14403v2",
          "12d262c682626c26a98bdafc7ec5dc6af3fae8247d26d4f2c712b1599dafce62", "CC0 1.0"),
    Paper("corrective_rag.pdf", "2401.15884v3",
          "975aa1fd3c1b603126e93ff99d6504858b61301bf1d34c9df88ebe53a0b026cb", "arXiv non-exclusive"),
    Paper("lost_in_the_middle.pdf", "2307.03172v3",
          "653b29619eae2ae4b361d7bdcdc06db4bedcb1b0eabe31814036beca8c1af0b1", "arXiv non-exclusive"),
    Paper("rag_lewis2020.pdf", "2005.11401v4",
          "23e3249e9a1e75418d82efecab0ea8c4d033b89c93742f63208d47ce01f21233", "arXiv non-exclusive"),
    Paper("react.pdf", "2210.03629v3",
          "f285b0971ae4a790e402fb93966bed3adde2cf0a04977d08b2b40d6ab0cace69", "CC BY 4.0"),
    Paper("self_rag.pdf", "2310.11511v1",
          "d9eaa1398abac0df67a9d0933a5bf8b6d9d83d2e72da2a486073cd842dd52978", "CC BY 4.0"),
]

# arXiv asks automated clients to wait between requests.
POLITE_DELAY_S = 3


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for paper in CORPUS:
        path = DOCS_DIR / paper.file
        if path.exists() and _sha256(path.read_bytes()) == paper.sha256:
            print(f"ok        {paper.file}")
            continue
        request = urllib.request.Request(
            f"https://arxiv.org/pdf/{paper.arxiv}",
            headers={"User-Agent": "agentic-rag (research demo)"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        if _sha256(data) != paper.sha256:
            sys.exit(f"{paper.file}: downloaded file doesn't match the pinned hash")
        path.write_bytes(data)
        print(f"download  {paper.file}  ({paper.arxiv}, {len(data) // 1024} KB)")
        time.sleep(POLITE_DELAY_S)


if __name__ == "__main__":
    download()
