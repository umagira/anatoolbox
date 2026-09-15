"""rewrite_query_for_retrieval — turn a question into retrieval-ready queries.

Runs before retrieval and never answers the question. Three strategies:

* ``expand`` — several phrasings and aspects of a broad question. A question
  like "What should we know about AI agents?" retrieves a little of everything;
  targeted variants retrieve evidence for each angle.
* ``decompose`` — one self-contained query per part of a compound question
  ("How do Nvidia's and AMD's new accelerators compare?"), so no part goes
  unretrieved.
* ``clarify`` — one precise query that resolves vague terms while keeping
  names, products and dates.

The queries feed ``retrieve_passages(query=..., queries=[...])``, which ranks
each one and fuses the rankings. ``exact_terms`` lists rare names and acronyms
worth matching literally — the case where keyword search beats embeddings — and
``time_range`` captures a period the question names, ready for ``date_from`` /
``date_to``. Small models like to fill it in from the collection's own time span,
so a range is only kept when the question itself mentions a period.

The model comes from the ``fast`` role of ``anatoolbox.llm_client``, falling back
to the default model. If the model returns no usable query, the original
question is used and ``fallback`` says so.

**Variants.** Subclass ``RewriteQueryForRetrievalTool`` and override
``system_prompt`` / ``user_prompt`` to change the instructions, ``rewrite`` to
produce queries another way (for example a hypothetical answer to embed
instead of the question), or ``clean`` to post-process the reply.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, ClassVar

from anatoolbox.base import ToolContext
from anatoolbox.gather.rewrite.base import PREFIX, STAGE
from anatoolbox.llm_client import call_llm_json, model_for
from anatoolbox.memory import value_ref
from anatoolbox.tool import BaseTool

TOOL_NAME = "rewrite_query_for_retrieval"
OBJECT_TYPE = "queries"
STRATEGIES = ("expand", "decompose", "clarify")
DEFAULT_STRATEGY = "expand"
DEFAULT_MAX_QUERIES = 3
MAX_QUERIES_LIMIT = 8
MAX_EXACT_TERMS = 10

# A year, quarter, month name, or a period phrase. "may" is left out on purpose:
# it is far more often a verb than a month, and "May 2025" matches on the year.
_PERIOD_HINT = re.compile(
    r"\b(?:19|20)\d{2}\b"
    r"|\bq[1-4]\b"
    r"|\b(?:january|february|march|april|june|july|august|september|october|november|december)\b"
    r"|\b(?:since|until|before|after|between|during)\b"
    r"|\b(?:last|this|past|previous)\s+(?:\d+\s+)?(?:days?|weeks?|months?|quarters?|years?)\b",
    re.IGNORECASE,
)


def question_names_a_period(question: str) -> bool:
    """Whether a question mentions a time period a date range could come from."""
    return bool(_PERIOD_HINT.search(question or ""))


STRATEGY_INSTRUCTIONS = {
    "expand": "Write up to {n} queries that cover different phrasings and aspects of the question, the most important first.",
    "decompose": (
        "If the question has several parts, write one self-contained query per part, up to {n}. "
        "If it has only one part, write a single query."
    ),
    "clarify": "Write one precise, self-contained query that resolves vague terms while keeping every name, product and date.",
}

SYSTEM_PROMPT = """You turn a user's question into search queries for a document collection. You never answer the question.

{strategy}

Rules:
- Keep names of companies, products, models and standards exactly as the user wrote them.
- Every query must stand on its own, without pronouns that point back at the question.
- purpose: a few words on what the query is meant to find.
- exact_terms: rare names, acronyms or version numbers worth matching literally; an empty list if there are none.
- time_range: ISO dates (YYYY-MM-DD) only when the question itself names a period — never the period the collection covers; otherwise null for both.{collection}"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["queries", "exact_terms", "time_range"],
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query", "purpose"],
                "properties": {"query": {"type": "string"}, "purpose": {"type": "string"}},
            },
        },
        "exact_terms": {"type": "array", "items": {"type": "string"}},
        "time_range": {
            "type": "object",
            "additionalProperties": False,
            "required": ["from", "to"],
            "properties": {
                "from": {"type": ["string", "null"]},
                "to": {"type": ["string", "null"]},
            },
        },
    },
}


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10]).isoformat()
    except ValueError:
        return None


def clean_reply(reply: Any, question: str, max_queries: int) -> dict[str, Any]:
    """Validate a model reply: unique non-empty queries, capped; tidy terms; ISO dates only.

    A date range is discarded when it is inverted, or when the question names
    no period at all — the model then invented it, typically from the
    collection's own time span.
    """
    reply = reply if isinstance(reply, dict) else {}
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in reply.get("queries") or []:
        if isinstance(item, str):
            text, purpose = item, ""
        elif isinstance(item, dict):
            text, purpose = str(item.get("query") or ""), str(item.get("purpose") or "")
        else:
            continue
        text = text.strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        queries.append({"query": text, "purpose": purpose.strip()})
        if len(queries) >= max_queries:
            break
    fallback = not queries
    if fallback:
        queries = [
            {
                "query": question,
                "purpose": "the original question; the model returned no usable query",
            }
        ]

    terms: list[str] = []
    for term in reply.get("exact_terms") or []:
        term = str(term).strip()
        if term and term not in terms:
            terms.append(term)
    time_range = reply.get("time_range") if isinstance(reply.get("time_range"), dict) else {}
    start, end = _iso_date(time_range.get("from")), _iso_date(time_range.get("to"))
    if (start and end and start > end) or not question_names_a_period(question):
        start = end = None
    return {
        "queries": queries,
        "exact_terms": terms[:MAX_EXACT_TERMS],
        "time_range": {"from": start, "to": end},
        "fallback": fallback,
    }


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The user's question."},
        "strategy": {
            "type": "string",
            "enum": list(STRATEGIES),
            "default": DEFAULT_STRATEGY,
            "description": (
                "expand = several phrasings and aspects; decompose = one query per part of a "
                "compound question; clarify = one precise query."
            ),
        },
        "max_queries": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_QUERIES_LIMIT,
            "default": DEFAULT_MAX_QUERIES,
        },
        "collection": {
            "type": "string",
            "description": "What the document collection covers, e.g. 'AI news articles, Sep 2024 – Aug 2025'.",
        },
        "model": {
            "type": "string",
            "description": "Model name; defaults to the configured fast role.",
        },
    },
    "required": ["question"],
}


class RewriteQueryForRetrievalTool(BaseTool):
    """Rewrite a question into queries for retrieval: expand, decompose, or clarify.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``system_prompt(settings)`` / ``user_prompt(settings)`` — the instructions.
    * ``rewrite(settings, context)`` — produce the raw reply (default: one JSON LLM call).
    * ``clean(reply, settings)`` — turn the reply into queries, exact terms and a time range.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Rewrite a question into retrieval queries before searching: expand a broad question "
        "into variants, decompose a compound one into parts, or clarify a vague one. Returns "
        "queries for retrieve_passages(queries=...), exact terms, and any time range the "
        "question names. Never answers the question."
    )
    input_schema = _INPUT_SCHEMA
    #: Model role used when no ``model`` is passed; see ``llm_client.configure_llm``.
    role: ClassVar[str] = "fast"
    #: Strategies ``settings`` accepts. A subclass adding one extends this and the prompt.
    strategies: ClassVar[tuple[str, ...]] = STRATEGIES

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments. Recorded in provenance, including the model actually used."""
        question = self.text_arg(args, "question", required=True)
        strategy = self.choice_arg(args, "strategy", DEFAULT_STRATEGY, self.strategies)
        max_queries = args.get("max_queries", DEFAULT_MAX_QUERIES)
        if (
            isinstance(max_queries, bool)
            or not isinstance(max_queries, int)
            or not 1 <= max_queries <= MAX_QUERIES_LIMIT
        ):
            raise self.input_error(
                f"max_queries must be an integer from 1 to {MAX_QUERIES_LIMIT}, got {max_queries!r}.",
                argument="max_queries",
                value=max_queries,
            )
        return {
            "question": question,
            "strategy": strategy,
            "max_queries": 1 if strategy == "clarify" else max_queries,
            "collection": self.text_arg(args, "collection") or None,
            "model": self.text_arg(args, "model") or model_for(self.role),
        }

    def system_prompt(self, settings: dict[str, Any]) -> str:
        collection = settings["collection"]
        return SYSTEM_PROMPT.format(
            strategy=STRATEGY_INSTRUCTIONS[settings["strategy"]].format(n=settings["max_queries"]),
            collection=f"\n- The collection covers: {collection}" if collection else "",
        )

    def user_prompt(self, settings: dict[str, Any]) -> str:
        return f"Question: {settings['question']}"

    def rewrite(self, settings: dict[str, Any], context: ToolContext) -> Any:
        """The raw reply: a dict with ``queries`` (strings or {query, purpose}), and
        optionally ``exact_terms`` and ``time_range``. Default: one JSON LLM call."""
        return call_llm_json(
            self.system_prompt(settings),
            self.user_prompt(settings),
            model=settings["model"],
            temperature=0.0,
            response_schema=RESPONSE_SCHEMA,
            project=context.project,
        )

    def clean(self, reply: Any, settings: dict[str, Any]) -> dict[str, Any]:
        """Validated ``queries``, ``exact_terms``, ``time_range`` and ``fallback``."""
        return clean_reply(reply, settings["question"], settings["max_queries"])

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        cleaned = self.clean(self.rewrite(settings, context), settings)
        result = {
            "question": settings["question"],
            "strategy": settings["strategy"],
            **cleaned,
            "query_texts": [q["query"] for q in cleaned["queries"]],
            "model": settings["model"],
        }
        result["handle"] = self._remember(context, result, max_queries=settings["max_queries"])
        result["provenance"] = self.provenance(settings)
        return result

    def _remember(
        self, context: ToolContext, result: dict[str, Any], *, max_queries: int
    ) -> str | None:
        if context.recordsets is None:
            return None
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=self.stage,
            produced_by=self.tool_name,
            args={
                "question": result["question"],
                "strategy": result["strategy"],
                "max_queries": max_queries,
                "model": result["model"],
            },
            ref=value_ref(result["queries"]),
            count=len(result["queries"]),
            summary=f"{len(result['queries'])} {result['strategy']} queries for {result['question']!r}",
            derived_from=[],
        )
        return record.handle
