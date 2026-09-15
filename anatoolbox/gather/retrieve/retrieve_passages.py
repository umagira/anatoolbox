"""retrieve_passages — rank a local corpus for a query.

One tool, three retrieval strategies, so comparing them is a changed argument
rather than a changed pipeline:

    retrieve_passages(query="agentic web standards", input=chunks, strategy="sparse")  # BM25
    retrieve_passages(query="agentic web standards", input=chunks, strategy="dense")   # embeddings
    retrieve_passages(query="agentic web standards", input=chunks, strategy="hybrid")  # both, fused

``input`` is the result of ``ingest_corpus`` or ``chunk_by_size``; its run id goes
into this result's provenance. ``size`` is how many passages come back — the top
k: too few misses evidence, too many adds noise and cost further down.

**Metadata.** ``filters`` keep only records whose metadata matches
(``{"domain": ["venturebeat"]}``; list fields such as tags match on any shared
value), and ``date_from`` / ``date_to`` keep a period. ``recency_half_life_days``
multiplies each score by ``0.5 ** (age / half_life)``, so newer records rise. Age
is measured from ``recency_reference_date``, which defaults to the newest date in
the corpus, so a historical corpus is not penalized just for being old.

**Several phrasings.** ``queries`` adds rewrites of the question, and
``queries_input`` takes the result of ``rewrite_query_for_retrieval``. Each query
is ranked on its own and the rankings are fused with reciprocal rank fusion, so a
document relevant to any phrasing can surface.

**Agents.** With recordset memory, ``input`` may instead be a handle such as
``corpus_1``, and without ``input`` the newest corpus is used.

**Other retrieval methods.** Subclass ``RetrievePassagesTool`` with a new
``tool_name`` and override a hook: ``rank`` (how one query ranks the corpus —
a different scoring function, another index, a weighted fusion), ``keep``
(which records may be returned), ``boost`` (a score multiplier per record) or
``fuse`` (how the rankings of several queries combine).
"""

from __future__ import annotations

from datetime import date
from typing import Any, ClassVar

from anatoolbox.base import ToolContext
from anatoolbox.corpus import (
    LocalCorpus,
    bind_corpus,
    ensure_bm25,
    ensure_embeddings,
    get_embedder,
    local_ref,
)
from anatoolbox.gather.retrieve.base import PREFIX, STAGE
from anatoolbox.provenance import run_id_of
from anatoolbox.retrieval import STRATEGIES, Hit, rank_records, reciprocal_rank_fusion
from anatoolbox.tool import BaseTool

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
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Additional phrasings of the query, e.g. from rewrite_query_for_retrieval. "
                "Each is ranked separately and the rankings are fused."
            ),
        },
        "queries_input": {
            "type": "string",
            "description": (
                "Handle of stored queries from rewrite_query_for_retrieval, e.g. 'queries_1'. "
                "They are fused like `queries`, and the lineage records where they came from."
            ),
        },
        "filters": {
            "type": "object",
            "description": (
                "Only records whose metadata matches: field name -> value or list of values. "
                "List fields such as tags match on any shared value."
            ),
        },
        "recency_half_life_days": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "Favor newer records: a record this many days older scores half as much.",
        },
        "recency_reference_date": {
            "type": "string",
            "description": "ISO date that ages are measured from (default: newest date in the corpus).",
        },
    },
    "required": ["query"],
}


class RetrievePassagesTool(BaseTool):
    """Rank a local corpus with BM25, embeddings, or a fusion of both.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``rank(query, corpus, settings, limit)`` — rank the corpus for one query.
    * ``keep(record, settings)`` — whether a record may be returned.
    * ``boost(record, settings)`` — a score multiplier per record.
    * ``fuse(rankings, settings)`` — combine the rankings of several queries.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Search a local corpus and return the best-matching passages. "
        "strategy=sparse|dense|hybrid selects BM25, embeddings, or rank fusion."
    )
    input_schema = _INPUT_SCHEMA
    render_type = "table"
    #: Strategies ``settings`` accepts. A subclass adding one extends this and ``rank``.
    strategies: ClassVar[tuple[str, ...]] = STRATEGIES

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments that shape the ranking. Recorded in provenance."""
        strategy = self.text_arg(args, "strategy") or DEFAULT_STRATEGY
        if strategy not in self.strategies:
            raise self.input_error(
                f"Unknown strategy {strategy!r}. Choose from {list(self.strategies)}.",
                argument="strategy",
                value=strategy,
            )
        return {
            "query": self.text_arg(args, "query", required=True),
            "strategy": strategy,
            "size": self.int_arg(args, "size", DEFAULT_SIZE, minimum=1),
            "date_field": self.text_arg(args, "date_field") or DEFAULT_DATE_FIELD,
            "date_from": self._date_arg(args, "date_from"),
            "date_to": self._date_arg(args, "date_to"),
            "filters": self._filters_arg(args) or None,
            "recency_half_life_days": self._half_life_arg(args),
            "recency_reference_date": self._date_arg(args, "recency_reference_date"),
        }

    def rank(
        self, query: str, corpus: LocalCorpus, settings: dict[str, Any], limit: int
    ) -> list[Hit]:
        """Up to ``limit`` hits for one query, best first. Override to rank differently.

        A ``Hit`` is ``(index, score, strategy)``, where ``index`` is the position of
        the record in ``corpus.records``. Apply ``keep`` and ``boost`` yourself if an
        override should still honour filters, dates and recency.
        """
        strategy = settings["strategy"]
        dense = strategy in ("dense", "hybrid")
        return rank_records(
            query=query,
            records=corpus.records,
            texts=corpus.texts(),
            strategy=strategy,
            size=limit,
            embedder=get_embedder() if dense else None,
            document_matrix=ensure_embeddings(corpus) if dense else None,
            bm25=ensure_bm25(corpus) if strategy in ("sparse", "hybrid") else None,
            predicate=(lambda record: self.keep(record, settings))
            if self._filters_records(settings)
            else None,
            boost=(lambda record: self.boost(record, settings))
            if self._boosts_records(settings)
            else None,
        )

    def keep(self, record: dict[str, Any], settings: dict[str, Any]) -> bool:
        """Whether ``record`` may be returned. Default: the date range and ``filters``."""
        date_from, date_to = settings["date_from"], settings["date_to"]
        if date_from or date_to:
            # A record of unknown date cannot be shown to be in range, and a
            # temporal question should not be answered from it.
            published = _published(record, settings["date_field"])
            if published is None:
                return False
            if date_from and published.isoformat() < date_from:
                return False
            if date_to and published.isoformat() > date_to:
                return False
        for field, allowed in (settings["filters"] or {}).items():
            value = record.get(field)
            if isinstance(value, (list, tuple, set)):
                present = {str(v) for v in value}
            else:
                present = set() if value is None else {str(value)}
            if not present & {str(v) for v in allowed}:
                return False
        return True

    def boost(self, record: dict[str, Any], settings: dict[str, Any]) -> float:
        """Score multiplier for ``record``. Default: recency decay ``0.5 ** (age / half_life)``.

        A record dated after the reference date counts as age 0. A record with no
        usable date gets 0: once a question asks for recency, evidence of unknown
        date should not outrank dated evidence.
        """
        half_life = settings["recency_half_life_days"]
        reference = settings["recency_reference_date"]
        if not half_life or not reference:
            return 1.0
        published = _published(record, settings["date_field"])
        if published is None:
            return 0.0
        return 0.5 ** (max(0, (date.fromisoformat(reference) - published).days) / half_life)

    def fuse(self, rankings: list[list[Hit]], settings: dict[str, Any]) -> list[Hit]:
        """One ranking from the rankings of several queries. Default: reciprocal rank fusion."""
        return reciprocal_rank_fusion(rankings)

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        query, size = settings["query"], settings["size"]
        extra_queries, queries_handle, queries_ref = self._extra_queries(args, context=context)
        all_queries = _unique_queries([query, *extra_queries])
        settings["queries"] = all_queries if len(all_queries) > 1 else None

        corpus, source_ref = bind_corpus(args, context, tool_name=self.tool_name)
        source_handle = None if isinstance(args.get("input"), dict) else source_ref
        if settings["recency_half_life_days"] and not settings["recency_reference_date"]:
            latest = _latest_date(corpus.records, settings["date_field"])
            settings["recency_reference_date"] = latest.isoformat() if latest else None

        if len(all_queries) == 1:
            hits = self.rank(query, corpus, settings, size)
        else:
            # Rank each phrasing deeper than `size`, then fuse.
            pool = max(size * 3, 30)
            rankings = [self.rank(text, corpus, settings, pool) for text in all_queries]
            hits = self.fuse(rankings, settings)
        hits = hits[:size]

        records, texts = corpus.records, corpus.texts()
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
            "query": query,
            "strategy": settings["strategy"],
            "corpus": corpus.name,
            "size": size,
            "returned": len(passages),
            "passages": passages,
            "input_handle": source_handle,
            "date_from": settings["date_from"],
            "date_to": settings["date_to"],
            "filters": settings["filters"],
            "queries": settings["queries"],
            "queries_input_handle": queries_handle,
            "recency_half_life_days": settings["recency_half_life_days"],
            "recency_reference_date": (
                settings["recency_reference_date"] if settings["recency_half_life_days"] else None
            ),
        }
        result["handle"] = self._remember(
            context,
            corpus=corpus,
            passages=passages,
            settings=settings,
            source_handle=source_handle,
            queries_handle=queries_handle,
        )
        result["provenance"] = self.provenance(
            {
                **settings,
                "corpus": corpus.name,
                "recency_reference_date": result["recency_reference_date"],
            },
            derived_from=[source_ref, queries_ref],
        )
        return result

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        columns = ["rank", "score", "id", "snippet"]
        return {
            "render_type": "table",
            "columns": columns,
            "rows": [{k: p.get(k) for k in columns} for p in data["passages"]],
            "caption": f"{data['returned']} passages for {data['query']!r} ({data['strategy']})",
        }

    # --- plumbing ----------------------------------------------------------

    def _filters_records(self, settings: dict[str, Any]) -> bool:
        """Whether ``keep`` has anything to decide; skipping it keeps an unfiltered walk fast."""
        overridden = type(self).keep is not RetrievePassagesTool.keep
        return overridden or bool(
            settings["date_from"] or settings["date_to"] or settings["filters"]
        )

    def _boosts_records(self, settings: dict[str, Any]) -> bool:
        """Whether ``boost`` can change anything; boosting re-sorts the whole ranking."""
        overridden = type(self).boost is not RetrievePassagesTool.boost
        return overridden or bool(
            settings["recency_half_life_days"] and settings["recency_reference_date"]
        )

    def _extra_queries(self, args: dict[str, Any], *, context: ToolContext):
        """``queries`` plus stored or upstream rewrites. Returns (texts, handle, provenance ref)."""
        extra = self._queries_arg(args)
        given = args.get("queries_input")
        if isinstance(given, dict):
            stored = [str(t).strip() for t in given.get("query_texts") or [] if str(t).strip()]
            if not stored:
                raise self.input_error(
                    "queries_input must be a queries handle or the result of "
                    "rewrite_query_for_retrieval.",
                    argument="queries_input",
                )
            return [*extra, *stored], None, run_id_of(given)
        if isinstance(given, str) and given.strip():
            if context.recordsets is None:
                raise self.input_error(
                    "queries_input needs recordset memory; pass the query texts as queries=[...] instead.",
                    code="missing_required_arguments",
                    argument="queries_input",
                )
            record = context.recordsets.bind(
                object_type="queries", tool_name=self.tool_name, requested=given.strip()
            )
            stored = []
            for row in context.recordsets.records(record):
                text = str(row.get("query") if isinstance(row, dict) else row).strip()
                if text:
                    stored.append(text)
            return [*extra, *stored], record.handle, record.handle
        return extra, None, None

    def _remember(
        self, context: ToolContext, *, corpus, passages, settings, source_handle, queries_handle
    ) -> str | None:
        """Store the hit ids by reference, with lineage back to the corpus."""
        if context.recordsets is None:
            return None
        args = {"query": settings["query"], "strategy": settings["strategy"], "corpus": corpus.name}
        args.update(
            {
                key: settings[key]
                for key in (
                    "date_from",
                    "date_to",
                    "filters",
                    "queries",
                    "recency_half_life_days",
                    "recency_reference_date",
                )
                if settings.get(key)
                and (key != "recency_reference_date" or settings["recency_half_life_days"])
            }
        )
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=self.stage,
            produced_by=self.tool_name,
            args=args,
            ref=local_ref(corpus=corpus.name, ids=[p["id"] for p in passages]),
            count=len(passages),
            summary=f"{len(passages)} passages for {settings['query']!r} ({settings['strategy']})",
            derived_from=[h for h in (source_handle, queries_handle) if h],
        )
        return record.handle

    def _date_arg(self, args: dict[str, Any], name: str) -> str | None:
        """An ISO date argument, normalized to YYYY-MM-DD; None when blank."""
        value = args.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return date.fromisoformat(str(value).strip()[:10]).isoformat()
        except ValueError as exc:
            raise self.input_error(
                f"{name} must be an ISO date (YYYY-MM-DD), got {value!r}.",
                argument=name,
                value=value,
            ) from exc

    def _half_life_arg(self, args: dict[str, Any]) -> float | None:
        value = args.get("recency_half_life_days")
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            days = float(value)
        except (TypeError, ValueError):
            days = 0.0
        if isinstance(value, bool) or days <= 0:
            raise self.input_error(
                f"recency_half_life_days must be a positive number of days, got {value!r}.",
                argument="recency_half_life_days",
                value=value,
            )
        return days

    def _filters_arg(self, args: dict[str, Any]) -> dict[str, list]:
        value = args.get("filters")
        if value is None or value == {}:
            return {}
        if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
            raise self.input_error(
                "filters must be an object mapping field names to a value or a list of values.",
                argument="filters",
                value=value,
            )
        return {
            field: list(values) if isinstance(values, (list, tuple, set)) else [values]
            for field, values in value.items()
        }

    def _queries_arg(self, args: dict[str, Any]) -> list[str]:
        value = args.get("queries")
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(q, str) for q in value):
            raise self.input_error(
                "queries must be a list of strings.", argument="queries", value=value
            )
        return [q.strip() for q in value if q.strip()]


def _published(record: dict, field: str) -> date | None:
    try:
        return date.fromisoformat(str(record.get(field)).strip()[:10])
    except ValueError:
        return None


def _latest_date(records: list[dict], field: str) -> date | None:
    dates = [d for d in (_published(r, field) for r in records) if d is not None]
    return max(dates) if dates else None


def _unique_queries(texts: list[str]) -> list[str]:
    """Drop repeated phrasings, ignoring case, keeping the first occurrence."""
    seen: set[str] = set()
    unique = []
    for text in texts:
        if text.casefold() not in seen:
            seen.add(text.casefold())
            unique.append(text)
    return unique
