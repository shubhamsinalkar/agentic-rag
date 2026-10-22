"""PDF -> page Documents -> chunks -> embeddings -> Chroma on disk.

Rebuild from the shipped vectors (free):  .venv\\Scripts\\python -m app.ingest --from-export
Rebuild by calling the API (635 requests): .venv\\Scripts\\python -m app.ingest
Search it:  .venv\\Scripts\\python -m app.ingest --search "your question"
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from google.genai.types import HttpOptions
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.query_cache import CachedQueryEmbeddings

load_dotenv()  # puts GOOGLE_API_KEY from .env into the environment

# ---------------------------------------------------------------- settings ---

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "papers"
QUERY_CACHE = ROOT / "cache" / "query_embeddings.sqlite"
INDEX_EXPORT = ROOT / "data" / "index"   # shipped vectors, see export_index()

# Newest stable Gemini embedding model. Unlike gemini-embedding-001, it still
# returns unit-length vectors when shortened (checked 2026-10-21).
EMBED_MODEL = "gemini-embedding-2"
# Full size is 3072. The model is trained so the first 768 numbers carry most
# of the meaning: 4x less disk/RAM for a small loss in retrieval quality.
EMBED_DIMS = 768
# A question embeds in ~0.7 s and a 90-chunk index batch in a few seconds;
# 30 s means the request is stuck, not slow.
EMBED_TIMEOUT_S = 30

# Measured in CHARACTERS, not tokens. 1000 chars ~ 250 tokens ~ one paragraph.
# 200 chars: one sentence, too little context to answer anything alone.
# 4000 chars: several topics averaged into one vector -> fuzzy matches.
CHUNK_SIZE = 1000
# A sentence cut at a chunk boundary still appears whole in one chunk.
# Cost: ~20% of the text is embedded twice.
CHUNK_OVERLAP = 200

# 4 chunks x ~250 tokens ~ 1k tokens of context for the LLM.
# Higher k = better chance the answer is in there, but more noise and cost.
TOP_K = 4

# Free tier caps embeddings per minute (roughly 100 texts / 30k tokens).
# First run with 60-chunk batches hit a 429 on every second batch, so one
# batch per minute is the real budget: 90 chunks ~ 22k tokens fits under both.
# A 429 (rate limited) waits a minute and retries instead of losing progress.
BATCH_CHUNKS = 90
RETRY_WAIT_S = 60
MAX_RETRIES = 5


# ------------------------------------------------------------------ steps ---

def load_pages() -> list[Document]:
    """Return one Document per PDF page, for every PDF in DOCS_DIR.

    A "loader" is just this: text + metadata wrapped in a Document. We write
    it ourselves instead of pulling in langchain-community (being retired).
    """
    pages = []
    for path in sorted(DOCS_DIR.glob("*.pdf")):
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""  # scanned pages have no text layer
            if not text.strip():
                continue
            # enumerate counts from 0; humans (and citations) count pages from 1
            pages.append(Document(page_content=text,
                                  metadata={"source": path.name, "page": i + 1}))
    return pages


def split(pages: list[Document]) -> list[Document]:
    """Split pages into overlapping chunks.

    split_documents (not split_text) copies each page's metadata onto every
    chunk cut from it, so every chunk still knows its paper and page.
    It tries to cut at paragraph breaks first, then lines, then words.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,  # records where in the page each chunk starts
    )
    return splitter.split_documents(pages)


def get_vectorstore() -> Chroma:
    """Open (or create) the Chroma collection on disk.

    Retrieval imports this too, so chunks and questions are always embedded
    by the same model. Mismatched models give garbage with no error.
    """
    google = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, output_dimensionality=EMBED_DIMS)
    # The wrapper has no working timeout setting (tested: its options are
    # ignored), so swap in a Google client that has one. Without it, a
    # stuck request waits forever.
    google.client = genai.Client(http_options=HttpOptions(timeout=EMBED_TIMEOUT_S * 1000))
    embeddings = CachedQueryEmbeddings(google, path=QUERY_CACHE,
                                       model_id=f"{EMBED_MODEL}@{EMBED_DIMS}")
    return Chroma(
        collection_name=COLLECTION,
        # Never omit this: without it Chroma silently downloads a local model.
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        # Chroma's default is L2 distance; cosine makes scores readable (0..2).
        collection_metadata={"hnsw:space": "cosine"},
    )


def chunk_id(doc: Document) -> str:
    """Stable id, e.g. "react.pdf:4:812" (paper, page, character offset).

    The same chunk gets the same id on every machine, which is what lets a
    saved index be matched back to freshly split PDFs.
    """
    m = doc.metadata
    return f"{m['source']}:{m['page']}:{m['start_index']}"


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_index() -> None:
    """Rebuild the whole index from scratch by calling the embedding API.

    Deleting first is what stops a second run from storing every chunk twice.
    The corpus is frozen and small, so a full rebuild is simpler than tracking
    which chunks changed.

    Cost: one embedding request PER CHUNK. 635 chunks use 635 of the free
    tier's 1,000 requests per day, so don't rebuild casually: deployments
    use build_from_export() instead, which costs nothing.
    """
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    pages = load_pages()
    print(f"loaded {len(pages)} pages")
    chunks = split(pages)
    print(f"split into {len(chunks)} chunks")
    started = time.perf_counter()
    add_in_batches(get_vectorstore(), chunks)
    print(f"done in {time.perf_counter() - started:.0f}s -> {CHROMA_DIR}")


# ------------------------------------------------- shipping the index ---
# The repo can't contain the papers' text (licenses), but embeddings are
# just numbers. So we ship the numbers plus a hash of each chunk's text;
# the deploy downloads the PDFs, re-splits them, checks every hash, and
# re-attaches the numbers. Same index, zero embedding API calls.

def export_index() -> None:
    """Save the current index's vectors + chunk hashes to data/index/."""
    got = get_vectorstore().get(include=["embeddings", "documents", "metadatas"])
    rows = sorted(zip(got["metadatas"], got["documents"], got["embeddings"]),
                  key=lambda r: (r[0]["source"], r[0]["page"], r[0]["start_index"]))
    INDEX_EXPORT.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_EXPORT / "embeddings.npy",
            np.array([r[2] for r in rows], dtype=np.float32))
    manifest = {
        "model": EMBED_MODEL, "dims": EMBED_DIMS,
        "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
        "chunks": [{"id": chunk_id(Document(page_content="", metadata=m)),
                    "sha256": _text_hash(text)} for m, text, _ in rows],
    }
    (INDEX_EXPORT / "chunks.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"exported {len(rows)} vectors -> {INDEX_EXPORT}")


def build_from_export() -> None:
    """Rebuild the index from data/index/ + the PDFs, with no API calls."""
    manifest = json.loads((INDEX_EXPORT / "chunks.json").read_text(encoding="utf-8"))
    if (manifest["model"], manifest["dims"], manifest["chunk_size"], manifest["chunk_overlap"]) \
            != (EMBED_MODEL, EMBED_DIMS, CHUNK_SIZE, CHUNK_OVERLAP):
        sys.exit("data/index was built with different settings; rebuild with the API")
    vectors = np.load(INDEX_EXPORT / "embeddings.npy")
    saved = {c["id"]: (c["sha256"], vectors[i]) for i, c in enumerate(manifest["chunks"])}

    chunks = split(load_pages())
    fresh = {chunk_id(d): d for d in chunks}
    if fresh.keys() != saved.keys():
        sys.exit(f"chunks differ from the export ({len(fresh)} vs {len(saved)}): "
                 "are the PDFs the pinned versions? run python -m app.corpus")
    bad = [i for i, d in fresh.items() if _text_hash(d.page_content) != saved[i][0]]
    if bad:
        sys.exit(f"{len(bad)} chunks have different text than the export, e.g. {bad[0]}")

    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    store = get_vectorstore()  # creates the collection with the right settings
    ids = list(fresh)
    # Straight into the Chroma collection: LangChain's add_documents would
    # call the embedding API, which is exactly what we're avoiding.
    store._collection.add(
        ids=ids,
        embeddings=[saved[i][1].tolist() for i in ids],
        documents=[fresh[i].page_content for i in ids],
        metadatas=[fresh[i].metadata for i in ids],
    )
    print(f"rebuilt {len(ids)} chunks from {INDEX_EXPORT} (0 API calls) -> {CHROMA_DIR}")


def search(question: str) -> None:
    """Print the TOP_K closest chunks for a question."""
    results = get_vectorstore().similarity_search_with_score(question, k=TOP_K)
    for rank, (doc, distance) in enumerate(results, start=1):
        # cosine DISTANCE = 1 - similarity. Lower = closer. Close != correct.
        print(f"\n#{rank}  distance={distance:.3f}  "
              f"{doc.metadata['source']} p.{doc.metadata['page']}")
        print("   " + doc.page_content[:300].replace("\n", " ") + " ...")


# ------------------------------------------------------------- plumbing ------

def add_in_batches(store: Chroma, chunks: list[Document]) -> None:
    """Embed and store chunks a batch at a time, waiting out rate limits."""
    for start in range(0, len(chunks), BATCH_CHUNKS):
        batch = chunks[start:start + BATCH_CHUNKS]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                store.add_documents(batch, ids=[chunk_id(d) for d in batch])
                break
            except Exception as err:
                cause = err.__cause__
                rate_limited = isinstance(cause, ClientError) and cause.code == 429
                # Wi-Fi blips show up as httpx connection errors; worth a retry.
                network_blip = isinstance(cause, httpx.TransportError)
                if not (rate_limited or network_blip) or attempt == MAX_RETRIES:
                    raise
                reason = "rate limited" if rate_limited else "network error"
                print(f"  {reason}, waiting {RETRY_WAIT_S}s (attempt {attempt})")
                time.sleep(RETRY_WAIT_S)
        print(f"  stored {start + len(batch)}/{len(chunks)} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--search", metavar="QUESTION", help="print top-k chunks")
    group.add_argument("--export", action="store_true",
                       help="save vectors + chunk hashes to data/index/")
    group.add_argument("--from-export", action="store_true",
                       help="rebuild the index from data/index/ (no API calls)")
    args = parser.parse_args()
    if args.search:
        search(args.search)
    elif args.export:
        export_index()
    elif args.from_export:
        build_from_export()
    else:
        build_index()
