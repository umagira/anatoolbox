"""retrieve_passages — rank a local corpus for a query.

One tool, three retrieval strategies, so comparing them is a changed argument
rather than a changed pipeline:

    retrieve_passages(query="agentic web standards", strategy="sparse")
    retrieve_passages(query="agentic web standards", strategy="dense")
    retrieve_passages(query="agentic web standards", strategy="hybrid")

Binding follows the usual consumer rules: an explicit ``input`` handle wins,
otherwise the newest corpus in memory is used. The result is remembered as a
``passages`` recordset whose lineage names the corpus it came from, so a later
score or table can say exactly which evidence it rests on.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any, ClassVar

from anatoolbox.base import ToolContext, ToolSchema
from anatoolbox.corpus import ensure_bm25, ensure_embeddings, get_corpus, get_embedder, local_ref
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.retrieve.base import PREFIX, STAGE
from anatoolbox.retrieval import STRATEGIES, rank_records

TOOL_NAME = "retrieve_passages"
OBJECT_TYPE = "passages"
CORPUS_OBJECT_TYPE = "corpus"
DEFAULT_SIZE = 10
DEFAULT_STRATEGY = "sparse"
DEFAULT_DATE_FIELD = "date"
#: Characters of text echoed back per hit. The full body stays in the corpus.
SNIPPET_CHARS = 400

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to search for."},
        "strategy": {
            "type": "string",
            "enum": list(STRATEGIES),
            "description": (
                "sparse = BM25 keyword match; dense = embedding similarity; "
                "hybrid = reciprocal rank fusion of both."
            ),
            "default": DEFAULT_STRATEGY,
        },
        "size": {
            "type": "integer",
            "description": f"How many passages to return (default: {DEFAULT_SIZE}).",
            "minimum": 1,
            "default": DEFAULT_SIZE,
        },
        "corpus": {
            "type": "string",
            "description": "Name of a registered corpus. Omit to use the bound recordset.",
        },
        "input": {
            "type": "string",
            "description": (
                "Optional handle of a corpus to search, e.g. 'corpus_1'. "
                "Omit to use the most recent one."
            ),
        },
        "date_from": {
            "type": "string",
            "description": "Only passages published on or after this ISO date (YYYY-MM-DD).",
        },
        "date_to": {
            "type": "string",
            "description": "Only passages published on or before this ISO date (YYYY-MM-DD).",
        },
        "date_field": {
            "type": "string",
            "description": f"Record field holding the publication date (default: {DEFAULT_DATE_FIELD}).",
            "default": DEFAULT_DATE_FIELD,
        },
    },
    "required": ["query"],
}


class RetrievePassagesTool:
    """Rank a local corpus with BM25, embeddings, or a fusion of both."""

    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE
    tool_name: ClassVar[str] = TOOL_NAME

    schema = ToolSchema(
        name=TOOL_NAME,
        description=(
            "Search a local corpus and return the best-matching passages. "
            "strategy=sparse|dense|hybrid selects BM25, embeddings, or rank fusion."
        ),
        input_schema=_INPUT_SCHEMA,
        render_type="table",
    )

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolInputError(
                code="missing_required_arguments",
                message="Argument 'query' is required.",
                tool_name=TOOL_NAME,
                details={"missing": ["query"]},
            )
        strategy = (args.get("strategy") or DEFAULT_STRATEGY).strip()
        if strategy not in STRATEGIES:
            raise ToolInputError(
                code="invalid_argument_value",
                message=f"Unknown strategy {strategy!r}. Choose from {list(STRATEGIES)}.",
                tool_name=TOOL_NAME,
                details={"argument": "strategy", "value": strategy},
            )
        size = int(args.get("size") or DEFAULT_SIZE)
        date_field = (args.get("date_field") or DEFAULT_DATE_FIELD).strip()
        date_from = _parse_date(args.get("date_from"), "date_from")
        date_to = _parse_date(args.get("date_to"), "date_to")

        corpus, source_handle = self._resolve_corpus(args, context=context)
        records = corpus.records
        texts = corpus.texts()

        document_matrix = None
        if strategy in ("dense", "hybrid"):
            document_matrix = ensure_embeddings(corpus)

        hits = rank_records(
            query=query.strip(),
            records=records,
            texts=texts,
            strategy=strategy,
            size=size,
            embedder=get_embedder() if strategy in ("dense", "hybrid") else None,
            document_matrix=document_matrix,
            bm25=ensure_bm25(corpus) if strategy in ("sparse", "hybrid") else None,
            predicate=_date_predicate(date_field, date_from, date_to),
        )

        passages = [
            {
                "id": str(records[hit.index].get(corpus.id_field)),
                "score": round(hit.score, 6),
                "rank": position + 1,
                "snippet": texts[hit.index][:SNIPPET_CHARS],
                **{
                    k: v
                    for k, v in records[hit.index].items()
                    if k not in (corpus.text_field, corpus.id_field)
                },
            }
            for position, hit in enumerate(hits)
        ]

        result = {
            "query": query.strip(),
            "strategy": strategy,
            "corpus": corpus.name,
            "size": size,
            "returned": len(passages),
            "passages": passages,
            "input_handle": source_handle,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
        }
        result["handle"] = self._remember(
            context,
            corpus=corpus,
            passages=passages,
            query=query.strip(),
            strategy=strategy,
            source_handle=source_handle,
            date_range=(result["date_from"], result["date_to"]),
        )
        return result

    def _resolve_corpus(self, args: dict[str, Any], *, context: ToolContext):
        """Explicit corpus name wins; then a handle; then the newest corpus."""
        named = (args.get("corpus") or "").strip()
        if named:
            return get_corpus(named), None

        if context.recordsets is None:
            raise ToolInputError(
                code="missing_required_arguments",
                message=(
                    "No corpus given and no recordset memory available. "
                    "Pass corpus='<name>', or run ingest_corpus first."
                ),
                tool_name=TOOL_NAME,
                details={"missing": ["corpus"]},
            )
        requested = args.get("input")
        record = context.recordsets.bind(
            object_type=CORPUS_OBJECT_TYPE,
            tool_name=TOOL_NAME,
            requested=requested.strip()
            if isinstance(requested, str) and requested.strip()
            else None,
        )
        corpus_name = str(record.ref.get("corpus") or record.args.get("corpus") or "")
        return get_corpus(corpus_name), record.handle

    def _remember(
        self,
        context: ToolContext,
        *,
        corpus,
        passages: list[dict[str, Any]],
        query: str,
        strategy: str,
        source_handle: str | None,
        date_range: tuple = (None, None),
    ) -> str | None:
        """Store the hit ids by reference, with lineage back to the corpus."""
        if context.recordsets is None:
            return None
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=STAGE,
            produced_by=TOOL_NAME,
            args={
                "query": query,
                "strategy": strategy,
                "corpus": corpus.name,
                **({"date_from": date_range[0]} if date_range[0] else {}),
                **({"date_to": date_range[1]} if date_range[1] else {}),
            },
            ref=local_ref(corpus=corpus.name, ids=[p["id"] for p in passages]),
            count=len(passages),
            summary=f"{len(passages)} passages for {query!r} ({strategy})",
            derived_from=[source_handle] if source_handle else [],
        )
        return record.handle

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str:
        data = self.run(args, context=context)
        render = {
            "render_type": "table",
            "columns": ["rank", "score", "id", "snippet"],
            "rows": [
                {k: p.get(k) for k in ("rank", "score", "id", "snippet")} for p in data["passages"]
            ],
            "caption": f"{data['returned']} passages for {data['query']!r} ({data['strategy']})",
        }
        return json.dumps({"render": render, **data})


def _parse_date(value: Any, argument: str) -> date | None:
    """ISO date argument -> ``date``; blank -> None; anything else is an input error."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError as exc:
        raise ToolInputError(
            code="invalid_argument_value",
            message=f"{argument} must be an ISO date (YYYY-MM-DD), got {value!r}.",
            tool_name=TOOL_NAME,
            details={"argument": argument, "value": value},
        ) from exc


def _date_predicate(field: str, date_from: date | None, date_to: date | None):
    """Keep records whose ``field`` falls in the range. None when unconstrained.

    A record with a missing or unparseable date is excluded once a range is
    set: it cannot be shown to be in range, and a temporal question should
    not be answered from evidence of unknown date.
    """
    if date_from is None and date_to is None:
        return None

    def in_range(record: dict) -> bool:
        raw = record.get(field)
        try:
            published = date.fromisoformat(str(raw).strip()[:10])
        except ValueError:
            return False
        if date_from is not None and published < date_from:
            return False
        if date_to is not None and published > date_to:
            return False
        return True

    return in_range
