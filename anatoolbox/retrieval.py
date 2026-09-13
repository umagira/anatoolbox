"""Ranking over a local corpus: sparse, dense, and hybrid.

Three strategies, one interface, so a comparison is a changed argument rather
than a changed pipeline:

* ``"sparse"`` — Okapi BM25 over tokenized text. Pure Python, no dependencies.
* ``"dense"`` — cosine similarity over embeddings. Needs ``numpy`` and an
  Embedder (see ``anatoolbox.corpus``).
* ``"hybrid"`` — reciprocal rank fusion of the two rankings. No score
  normalization, so the two scales never have to be reconciled.

A note on filtering, learned the hard way: filters are applied while *walking*
the ranking, not by taking the top ``size`` and filtering afterwards. Slicing
first and filtering after returns three results for ``size=10`` as soon as a
filter is at all selective.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

BM25_K1 = 1.5
BM25_B = 0.75
#: Rank-fusion damping. 60 is the value from the original RRF paper.
RRF_K = 60

STRATEGIES = ("sparse", "dense", "hybrid")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Deliberately simple and easy to replace."""
    return TOKEN_RE.findall((text or "").lower())


@dataclass
class Hit:
    """One ranked record: where it came from, and why it scored."""

    index: int
    score: float
    strategy: str


# --- sparse ---------------------------------------------------------------


class BM25Index:
    """Okapi BM25. Built once per corpus, queried many times."""

    def __init__(self, texts: Sequence[str], *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter] = []
        self.doc_lengths: list[int] = []
        # term -> indices of documents containing it. Lets a query touch only
        # the documents that can score, instead of every document in the corpus.
        self.postings: dict[str, list[int]] = {}
        for index, text in enumerate(texts):
            tokens = tokenize(text)
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.doc_lengths.append(len(tokens))
            for term in frequencies:
                self.postings.setdefault(term, []).append(index)
        self.doc_count = len(self.term_frequencies)
        self.avg_length = (sum(self.doc_lengths) / self.doc_count) if self.doc_count else 0.0
        self.document_frequency = Counter({t: len(ids) for t, ids in self.postings.items()})

    def idf(self, term: str) -> float:
        df = self.document_frequency.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        frequencies = self.term_frequencies[doc_index]
        length = self.doc_lengths[doc_index]
        norm = self.k1 * (
            1 - self.b + self.b * (length / self.avg_length if self.avg_length else 0)
        )
        total = 0.0
        for term in query_tokens:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            total += self.idf(term) * (frequency * (self.k1 + 1)) / (frequency + norm)
        return total

    def rank(self, query: str) -> list[Hit]:
        """Every document with a non-zero score, best first."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        candidates: set[int] = set()
        for term in set(query_tokens):
            candidates.update(self.postings.get(term, ()))
        scored = [
            Hit(index=i, score=self.score(query_tokens, i), strategy="sparse") for i in candidates
        ]
        scored = [hit for hit in scored if hit.score > 0]
        scored.sort(key=lambda h: (-h.score, h.index))
        return scored


# --- dense ----------------------------------------------------------------


def _try_numpy() -> Any:
    """numpy if installed, else None. Dense ranking works either way."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - depends on the extra
        return None
    return np


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def dense_rank(query_vector: Any, document_matrix: Any) -> list[Hit]:
    """Cosine similarity of one query against every document, best first.

    Uses numpy when it is installed and falls back to plain Python when it is
    not, so dense retrieval is testable — and usable on a small corpus —
    without pulling in the whole numeric stack.

    Only positive similarities are returned, matching BM25's behaviour of
    omitting non-matching documents. Padding results out to ``size`` with
    zero-similarity passages would put irrelevant evidence in front of a
    downstream answer.
    """
    np = _try_numpy()
    query = query_vector[0] if _is_matrix(query_vector) else query_vector

    width = len(document_matrix[0]) if len(document_matrix) else len(query)
    if len(query) != width:
        raise ValueError(
            f"Query vector has {len(query)} dimensions but the document matrix has "
            f"{width} — these came from different embedding models."
        )

    if np is not None:
        documents = np.asarray(document_matrix, dtype="float64")
        vector = np.asarray(query, dtype="float64")
        doc_norms = np.linalg.norm(documents, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        query_norm = np.linalg.norm(vector) or 1.0
        scores = (documents @ vector) / (doc_norms * query_norm)
        order = np.argsort(-scores)
        return [
            Hit(index=int(i), score=float(scores[i]), strategy="dense")
            for i in order
            if scores[i] > 0
        ]

    scored = [
        Hit(index=i, score=_cosine(row, query), strategy="dense")
        for i, row in enumerate(document_matrix)
    ]
    scored = [hit for hit in scored if hit.score > 0]
    scored.sort(key=lambda h: (-h.score, h.index))
    return scored


def _is_matrix(value: Any) -> bool:
    """True when ``value`` is a sequence of vectors rather than one vector."""
    try:
        first = value[0]
    except (IndexError, KeyError, TypeError):
        return False
    return hasattr(first, "__len__") and not isinstance(first, (str, bytes))


# --- fusion ---------------------------------------------------------------


def reciprocal_rank_fusion(rankings: Iterable[Sequence[Hit]], *, k: int = RRF_K) -> list[Hit]:
    """Combine rankings by rank position rather than by score.

    Score scales differ wildly between BM25 and cosine similarity; ranks do
    not, which is the whole point of RRF.
    """
    totals: dict[int, float] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking):
            totals[hit.index] = totals.get(hit.index, 0.0) + 1.0 / (k + position + 1)
    fused = [Hit(index=i, score=s, strategy="hybrid") for i, s in totals.items()]
    fused.sort(key=lambda h: (-h.score, h.index))
    return fused


# --- the one entry point --------------------------------------------------

#: Decides whether a record may be returned. Applied while walking the
#: ranking, so a selective filter still fills ``size``.
Predicate = Callable[[dict], bool]


def rank_records(
    *,
    query: str,
    records: Sequence[dict],
    texts: Sequence[str],
    strategy: str = "sparse",
    size: int = 10,
    embedder: Callable[[list[str]], Any] | None = None,
    document_matrix: Any | None = None,
    predicate: Predicate | None = None,
    bm25: BM25Index | None = None,
) -> list[Hit]:
    """Rank ``records`` for ``query`` and return at most ``size`` allowed hits.

    ``document_matrix`` lets a caller reuse precomputed embeddings; without
    it, dense strategies embed ``texts`` on the spot.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from {STRATEGIES}.")
    if strategy in ("dense", "hybrid") and embedder is None and document_matrix is None:
        raise ValueError(
            f"strategy={strategy!r} needs an embedder (or a precomputed document_matrix)."
        )

    rankings: list[list[Hit]] = []
    if strategy in ("sparse", "hybrid"):
        index = bm25 or BM25Index(texts)
        rankings.append(index.rank(query))
    if strategy in ("dense", "hybrid"):
        matrix = document_matrix if document_matrix is not None else embedder(list(texts))
        query_vector = embedder([query]) if embedder is not None else None
        if query_vector is None:
            raise ValueError("Dense ranking needs an embedder to encode the query.")
        rankings.append(dense_rank(query_vector, matrix))

    ranking = rankings[0] if len(rankings) == 1 else reciprocal_rank_fusion(rankings)

    # Walk, don't slice: filtering after a slice silently under-fills.
    kept: list[Hit] = []
    for hit in ranking:
        if len(kept) >= size:
            break
        if predicate is not None and not predicate(records[hit.index]):
            continue
        kept.append(hit)
    return kept
