"""rerank_passages — re-score retrieved candidates on their full text.

Retrieval ranks compressed representations; this step reads each candidate's
original text together with the query and re-orders the candidates by that
judgment. A reranker is slow, so it runs on a generous candidate set from
retrieval rather than on the whole corpus:

    retrieve_passages(query="...", strategy="hybrid", size=30)   # fast, broad
    rerank_passages(keep=5)                                      # slow, precise

The reranker reads the **full text** of every candidate from its corpus — not
the retrieval snippet — because recovering what compression and truncation
lost is the point. ``max_chars`` bounds how much of each text it reads, and a
cross-encoder stops at its own token limit anyway (often 512 tokens): one more
reason to rerank chunks rather than whole articles.

Every result reports its ``retrieval_rank`` and ``rank_change``, so the effect
of reranking is visible rather than assumed. The default reranker is a small
cross-encoder; plug in any other with ``anatoolbox.reranking.configure_reranker``,
or subclass ``RerankPassagesTool`` and override ``score`` (for example, to let a
language model judge relevance).
"""

from __future__ import annotations

from typing import Any

from anatoolbox.base import ToolContext
from anatoolbox.corpus import get_corpus, local_ref, passages_with_text
from anatoolbox.gather.rerank.base import PREFIX, STAGE
from anatoolbox.memory import value_ref
from anatoolbox.provenance import run_id_of
from anatoolbox.reranking import rerank_texts, reranker_label
from anatoolbox.tool import BaseTool

TOOL_NAME = "rerank_passages"
OBJECT_TYPE = "passages"
DEFAULT_KEEP = 5
DEFAULT_MAX_CHARS = 4000
SNIPPET_CHARS = 400
#: Fields of an upstream passage that describe its old ranking, not the passage.
_RANKING_FIELDS = (
    "text",
    "snippet",
    "rank",
    "score",
    "rerank_score",
    "retrieval_rank",
    "rank_change",
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Query to rerank for. Omit to reuse the query the candidates were retrieved with.",
        },
        "input": {
            "type": "string",
            "description": "Handle of the candidate passages, e.g. 'passages_3'. Omit to use the most recent.",
        },
        "keep": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_KEEP,
            "description": "How many passages to keep after reranking.",
        },
        "max_per_source": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Keep at most this many passages from one article (by source_id), so chunks "
                "of a single article cannot fill every slot."
            ),
        },
        "max_chars": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_MAX_CHARS,
            "description": "Read at most this many characters of each candidate's text.",
        },
        "passages": {
            "type": "array",
            "items": {"type": "object"},
            "description": (
                "Optional explicit candidates with 'id' and full 'text', for pipelines. "
                "The agent should normally omit this and let the tool bind stored passages."
            ),
        },
    },
}


class RerankPassagesTool(BaseTool):
    """Re-order retrieved passages by a reranker's judgment of their full text.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``score(query, texts, settings)`` — one relevance score per candidate text.
    * ``reranker(settings)`` — the label recorded for the scoring method.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Rerank previously retrieved passages: re-score each candidate's full text "
        "against the query with a reranking model and keep the best. Retrieve a "
        "generous set first (e.g. size=30), then rerank to keep=5."
    )
    input_schema = _INPUT_SCHEMA
    render_type = "table"

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments. ``query`` may be empty: the candidates' own query is used then."""
        return {
            "query": self.text_arg(args, "query"),
            "keep": self.int_arg(args, "keep", DEFAULT_KEEP),
            "max_per_source": self.int_arg(args, "max_per_source", None),
            "max_chars": self.int_arg(args, "max_chars", DEFAULT_MAX_CHARS),
        }

    def score(self, query: str, texts: list[str], settings: dict[str, Any]) -> list[float]:
        """One score per text, higher is more relevant. Default: the configured reranker.

        ``texts`` are already cut to ``max_chars``.
        """
        return rerank_texts(query, texts)

    def reranker(self, settings: dict[str, Any]) -> str:
        """A label for the scoring method, recorded with every result."""
        return reranker_label()

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        keep, max_per_source = settings["keep"], settings["max_per_source"]
        candidates, corpus_name, candidate_handle, candidate_query, upstream_ref = self._candidates(
            args, context=context
        )
        query = settings["query"] or str(candidate_query or "").strip()
        if not query:
            raise self.input_error(
                "No query to rerank for: pass `query`, or rerank passages that were retrieved with one.",
                code="missing_required_arguments",
                missing=["query"],
            )
        if not candidates:
            raise self.input_error(
                "There are no candidate passages to rerank; retrieval returned nothing.",
                code="no_candidates",
                input_handle=candidate_handle,
            )
        settings["query"] = query

        scores = self.score(
            query, [text[: settings["max_chars"]] for _, text, _ in candidates], settings
        )
        if len(scores) != len(candidates):
            raise ValueError(
                f"{type(self).__name__}.score returned {len(scores)} scores for {len(candidates)} texts."
            )
        order: list[int] = []
        per_source: dict[str, int] = {}
        skipped = 0
        for i in sorted(range(len(candidates)), key=lambda i: (-scores[i], i)):
            if len(order) >= keep:
                break
            source_key = str(candidates[i][2].get("source_id") or candidates[i][0])
            if max_per_source is not None and per_source.get(source_key, 0) >= max_per_source:
                skipped += 1
                continue
            per_source[source_key] = per_source.get(source_key, 0) + 1
            order.append(i)
        passages = []
        for new_rank, i in enumerate(order, start=1):
            passage_id, text, metadata = candidates[i]
            passages.append(
                {
                    **metadata,
                    "id": passage_id,
                    "rank": new_rank,
                    "rerank_score": round(float(scores[i]), 6),
                    "retrieval_rank": i + 1,
                    "rank_change": (i + 1) - new_rank,
                    "snippet": text[:SNIPPET_CHARS],
                }
            )

        label = self.reranker(settings)
        result = {
            "query": query,
            "reranker": label,
            "corpus": corpus_name,
            "candidates": len(candidates),
            "returned": len(passages),
            "skipped_over_source_limit": skipped,
            "passages": passages,
            "input_handle": candidate_handle,
        }
        recorded = {**settings, "reranker": label, "candidates": len(candidates)}
        result["handle"] = self._remember(
            context,
            passages=passages,
            corpus_name=corpus_name,
            candidate_handle=candidate_handle,
            settings=recorded,
        )
        result["provenance"] = self.provenance(
            {**recorded, "corpus": corpus_name}, derived_from=[upstream_ref]
        )
        return result

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        columns = ["rank", "rerank_score", "retrieval_rank", "id", "snippet"]
        return {
            "render_type": "table",
            "columns": columns,
            "rows": [{k: p.get(k) for k in columns} for p in data["passages"]],
            "caption": (
                f"top {data['returned']} of {data['candidates']} candidates for "
                f"{data['query']!r}, reranked by {data['reranker']}"
            ),
        }

    def _candidates(self, args: dict[str, Any], *, context: ToolContext):
        """Where the candidates come from, in precedence order.

        1. ``passages`` — explicit passages (full text, or ids plus ``corpus``).
        2. ``input`` as a result object — the result of ``retrieve_passages``
           (pipelines): its passages, corpus and query. Full texts are read from
           the corpus, because results carry only snippets.
        3. ``input`` as a handle, or nothing — a stored passages recordset (agents).

        Returns ``(candidates, corpus_name, handle, query, upstream_ref)``, where
        candidates are ``(id, full_text, metadata)`` in their original order and
        ``upstream_ref`` is a run id or handle for provenance.
        """
        given = args.get("input")
        explicit = args.get("passages")
        query_hint = None
        upstream = None
        if isinstance(given, dict):
            if explicit is None:
                explicit = given.get("passages")
            if explicit is None:
                raise self.input_error(
                    "`input` must be a passages handle or a result with 'passages', "
                    "such as the result of retrieve_passages.",
                    argument="input",
                )
            query_hint = given.get("query")
            upstream = run_id_of(given)
        if explicit is not None:
            given_corpus = given.get("corpus") if isinstance(given, dict) else ""
            corpus_hint = str(args.get("corpus") or given_corpus or "").strip() or None
            passages = passages_with_text(explicit, corpus_hint, tool_name=self.tool_name)
            return (
                [
                    (p["id"], p["text"], {k: v for k, v in p.items() if k not in _RANKING_FIELDS})
                    for p in passages
                ],
                corpus_hint,
                None,
                query_hint,
                upstream,
            )

        if context.recordsets is None:
            raise self.input_error(
                "No candidates: pass passages=[...], or run retrieve_passages first.",
                code="missing_required_arguments",
                missing=["passages"],
            )
        requested = args.get("input")
        record = context.recordsets.bind(
            object_type=OBJECT_TYPE,
            tool_name=self.tool_name,
            requested=requested.strip()
            if isinstance(requested, str) and requested.strip()
            else None,
        )
        corpus_name = str(record.ref.get("corpus") or "")
        try:
            corpus = get_corpus(corpus_name)
        except KeyError as exc:
            raise self.input_error(
                str(exc.args[0]) if exc.args else f"No corpus named {corpus_name!r}.",
                code="unknown_corpus",
                corpus=corpus_name,
                input_handle=record.handle,
            ) from exc
        candidates = [
            (
                str(row.get(corpus.id_field)),
                corpus.text_of(row),
                {k: v for k, v in row.items() if k not in (corpus.text_field, corpus.id_field)},
            )
            for row in corpus.get(record.ref.get("ids") or [])
        ]
        return candidates, corpus_name, record.handle, record.args.get("query"), record.handle

    def _remember(
        self, context, *, passages, corpus_name, candidate_handle, settings
    ) -> str | None:
        if context.recordsets is None:
            return None
        ids = [p["id"] for p in passages]
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=self.stage,
            produced_by=self.tool_name,
            args=settings,
            # Candidates from a corpus stay by reference; explicit ones exist nowhere else.
            ref=local_ref(corpus=corpus_name, ids=ids) if corpus_name else value_ref(passages),
            count=len(passages),
            summary=(
                f"top {len(passages)} of {settings['candidates']} candidates reranked "
                f"for {settings['query']!r}"
            ),
            derived_from=[candidate_handle] if candidate_handle else [],
        )
        return record.handle
