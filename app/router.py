"""Query router: TF-IDF + corpus coverage -> logistic regression.

Trained and evaluated in notebooks/01_router.ipynb, saved to models/router.pkl.
Predicts one of: factual, conceptual, comparison, off_topic (~1 ms, no API).

Why the extra "coverage" feature: TF-IDF silently drops words it never saw
in training, so for "Who won the FIFA World Cup?" it only sees "who" and
"won". The words that make a question off-topic are exactly the ones it
can't see. Coverage measures what fraction of the question's content words
appear in the papers at all, which TF-IDF has no way to represent.

The custom transformer lives here (not in the notebook) because a pickle
stores a *reference* to the class: loading router.pkl needs
app.router.CorpusCoverage importable.
"""

import re
import warnings
from collections import Counter
from functools import lru_cache

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from app.ingest import ROOT, load_pages

MODEL_PATH = ROOT / "models" / "router.pkl"


def content_words(text: str) -> list[str]:
    """Lowercase words of 3+ letters, minus stop words.

    Stop words are noise HERE ("what" appears in every text, so it says
    nothing about topic) even though they are the main signal for the
    question TYPE. Same words, different job.
    """
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) >= 3 and w not in ENGLISH_STOP_WORDS]


@lru_cache(maxsize=1)
def _corpus_counts() -> Counter:
    return Counter(w for p in load_pages() for w in content_words(p.page_content))


class CorpusCoverage(BaseEstimator, TransformerMixin):
    """One feature per question: share of its content words found in the papers.

    min_count=3: a word must appear 3+ times to count as the papers'
    vocabulary; words seen once or twice are mostly one-off examples and
    reference titles (e.g. "Paris" from a sample question).
    """

    def __init__(self, min_count: int = 3):
        self.min_count = min_count

    def fit(self, X, y=None):
        # The vocabulary comes from the papers, not from the labelled
        # questions, so it doesn't depend on the training split.
        self.vocab_ = frozenset(w for w, c in _corpus_counts().items()
                                if c >= self.min_count)
        return self

    def transform(self, X):
        def coverage(q: str) -> float:
            words = content_words(q)
            return sum(w in self.vocab_ for w in words) / len(words) if words else 0.0
        return np.array([[coverage(q)] for q in X])

    def get_feature_names_out(self, input_features=None):
        # Lets FeatureUnion name the column, so its weight can be inspected.
        return np.array(["coverage"])


def choose_route(proba: np.ndarray, classes: list[str], off_topic_min_prob: float) -> str:
    """Top label, except: refuse as off_topic only when confident.

    Refusing an answerable question is a wrong answer; sending an off-topic
    one down the normal path only costs time (the grader then says
    "I don't know"). So an unsure off_topic falls back to the runner-up.
    """
    order = np.argsort(proba)[::-1]
    best = classes[order[0]]
    if best == "off_topic" and proba[order[0]] < off_topic_min_prob:
        best = classes[order[1]]
    return best


@lru_cache(maxsize=1)
def _bundle() -> dict:
    # Only load pickles you created yourself: unpickling can run code.
    bundle = joblib.load(MODEL_PATH)
    if bundle["sklearn_version"] != sklearn.__version__:
        warnings.warn(f"router.pkl was trained with scikit-learn "
                      f"{bundle['sklearn_version']}, running {sklearn.__version__}; "
                      "retrain with notebooks/01_router.ipynb")
    return bundle


def predict(question: str) -> str:
    bundle = _bundle()
    pipeline = bundle["pipeline"]
    proba = pipeline.predict_proba([question])[0]
    return choose_route(proba, list(pipeline.classes_), bundle["off_topic_min_prob"])
