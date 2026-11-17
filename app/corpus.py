"""The frozen corpus: six arXiv papers, pinned to exact versions.

The PDFs are NOT in the repository: four of them (CheXNet, Four Degrees,
Planet Nine, Transformer) carry arXiv's standard license, which lets arXiv
distribute them but not us. Anyone (and the Render build) fetches them from
arXiv instead:

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


# Six readable papers from six different fields. None is about RAG, so the
# corpus can't be confused with the system that searches it.
CORPUS = [
    Paper("chexnet.pdf", "1711.05225v3",              # medicine
          "3419d5eb50b8f33f41f2d47d0e727c6933a4918d36b861968e6c3854138d86be", "arXiv non-exclusive"),
    Paper("four_degrees.pdf", "1111.4570v3",          # social networks
          "d6580859cf0a6130b247337e52e6c66f596149674d5cbfbb6758aab258b7187d", "arXiv non-exclusive"),
    Paper("generative_ai_at_work.pdf", "2304.11771v2",  # economics
          "146dcc7459824f0f1fe4a1f9ac7e8be06ee098b8b34e686a75131b77e42ce951", "CC BY-NC-ND 4.0"),
    Paper("gravitational_waves.pdf", "1602.03837v1",  # physics
          "e5e864c23d015b69be17e5b5d51b5b462d2829353a867513414b6728f54589c4", "CC BY 4.0"),
    Paper("planet_nine.pdf", "1601.05438v1",          # astronomy
          "23c08971aa86ee16d32fca1be7c48d2e70ffd7974d58a012abf0c1f7c42ce059", "arXiv non-exclusive"),
    Paper("transformer.pdf", "1706.03762v7",          # computer science
          "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697", "arXiv non-exclusive"),
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
