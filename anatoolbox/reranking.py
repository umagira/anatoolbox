"""Rerankers — re-score retrieved candidates against the query on their full text.

Embedding retrieval ranks compressed representations of documents, and each
representation is a generic, query-independent summary of its document. Some
information is lost on the way, so the most relevant records do not reliably
reach the top. A reranker reads the query and each candidate's original text
together, at query time, and scores relevance in the context of the question.

Rerankers are slow where precomputed embeddings are fast, which decides how the
two are combined: retrieve a generous candidate set cheaply, then rerank it and
keep the best few.

A Reranker is any callable ``(query, texts) -> scores`` where a higher score
means more relevant. Scores only need to be comparable within one call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

Reranker = Callable[[str, list[str]], Sequence[float]]

#: A small cross-encoder trained for passage relevance (MS MARCO).
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_RERANKER: Reranker | None = None
_RERANKER_LABEL: str | None = None
_MODELS: dict[str, Any] = {}


def cross_encoder_reranker(
    model_name: str = DEFAULT_RERANK_MODEL, *, batch_size: int = 32
) -> Reranker:
    """A Reranker backed by a sentence-transformers cross-encoder (``embeddings`` extra)."""

    def rerank(query: str, texts: list[str]) -> list[float]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "Reranking with a cross-encoder needs sentence-transformers: "
                "pip install 'anatoolbox[embeddings]'"
            ) from exc
        model = _MODELS.get(model_name)
        if model is None:
            model = CrossEncoder(model_name)
            _MODELS[model_name] = model
        scores = model.predict(
            [(query, text) for text in texts], batch_size=batch_size, show_progress_bar=False
        )
        return [float(score) for score in scores]

    rerank.label = model_name  # type: ignore[attr-defined]
    return rerank


def configure_reranker(reranker: Reranker | None, *, label: str | None = None) -> None:
    """Set the process-wide Reranker; ``None`` restores the default cross-encoder.

    ``label`` names the reranker in provenance. Without it, the callable's
    ``label`` attribute or function name is used.
    """
    global _RERANKER, _RERANKER_LABEL
    _RERANKER = reranker
    _RERANKER_LABEL = label


def get_reranker() -> Reranker:
    return _RERANKER if _RERANKER is not None else cross_encoder_reranker()


def reranker_label() -> str:
    """How the active reranker is named in recordset provenance."""
    if _RERANKER is None:
        return DEFAULT_RERANK_MODEL
    if _RERANKER_LABEL:
        return _RERANKER_LABEL
    return str(getattr(_RERANKER, "label", None) or getattr(_RERANKER, "__name__", "custom"))


def rerank_texts(query: str, texts: list[str], reranker: Reranker | None = None) -> list[float]:
    """Score ``texts`` for ``query``, checking the reranker returned one score per text."""
    scores = list((reranker or get_reranker())(query, list(texts)))
    if len(scores) != len(texts):
        raise ValueError(
            f"Reranker returned {len(scores)} scores for {len(texts)} texts; "
            "it must return exactly one score per text."
        )
    return [float(score) for score in scores]
