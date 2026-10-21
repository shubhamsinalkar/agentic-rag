"""PDF -> page Documents -> chunks -> embeddings -> Chroma on disk.

Build the index:   .venv\\Scripts\\python -m app.ingest
Search it:         .venv\\Scripts\\python -m app.ingest --search "your question"
"""

import argparse
import shutil
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google.genai.errors import ClientError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()  # puts GOOGLE_API_KEY from .env into the environment

# ---------------------------------------------------------------- settings ---

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "papers"

# Newest stable Gemini embedding model. Unlike gemini-embedding-001, it still
# returns unit-length vectors when shortened (checked 2026-10-21).
EMBED_MODEL = "gemini-embedding-2"
# Full size is 3072. The model is trained so the first 768 numbers carry most
# of the meaning: 4x less disk/RAM for a small loss in retrieval quality.
EMBED_DIMS = 768

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
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL,
                                              output_dimensionality=EMBED_DIMS)
    return Chroma(
        collection_name=COLLECTION,
        # Never omit this: without it Chroma silently downloads a local model.
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        # Chroma's default is L2 distance; cosine makes scores readable (0..2).
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index() -> None:
    """Rebuild the whole index from scratch.

    Deleting first is what stops a second run from storing every chunk twice.
    The corpus is frozen and small, so a full rebuild is simpler than tracking
    which chunks changed.
    """
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    pages = load_pages()
    print(f"loaded {len(pages)} pages")
    chunks = split(pages)
    print(f"split into {len(chunks)} chunks")
    started = time.perf_counter()
    add_in_batches(get_vectorstore(), chunks)
    print(f"done in {time.perf_counter() - started:.0f}s -> {CHROMA_DIR}")


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
                store.add_documents(batch)
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
    parser.add_argument("--search", metavar="QUESTION", help="print top-k chunks")
    args = parser.parse_args()
    if args.search:
        search(args.search)
    else:
        build_index()
