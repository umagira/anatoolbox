"""calculate_retrieval_metrics — precision, recall, hit rate and MRR over a test set.

Quantitative search evaluation compares, per question, what was retrieved with
what should have been:

    calculate_retrieval_metrics(results=[
        {"question": q["question"], "relevant": q["source_ids"], "retrieved": retrieved},
        ...
    ], k=[1, 3, 5, 10])

For each question and each ``k``:

* **precision@k** — the share of the top k that is relevant. It is divided by
  k, so returning fewer than k results does not raise it.
* **recall@k** — the share of the relevant items found in the top k.
* **hit@k** — 1 if any relevant item is in the top k, else 0.

And the **reciprocal rank**: 1 / the rank of the first relevant item, 0 if none
was retrieved. Its mean over the questions is the MRR. ``summary`` averages
every metric over the questions (a question without relevant items has no
recall and is left out of that mean).

**Articles, not chunks.** Relevance is usually known per article, while
retrieval over a chunk corpus returns chunks. With ``match_on="source"`` (the
default) a passage counts as its article — its ``source_id``, else its ``id`` —
and further chunks of an article already seen are skipped, so five chunks of
one article take one rank, not five. ``match_on="id"`` compares ids as they are.

``retrieved`` may be the result of ``retrieve_passages`` or ``rerank_passages``
(its run id goes into the provenance), a list of passages, or a list of ids.

**Variants.** Subclass and override ``metrics`` to add a measure (nDCG,
average precision), or ``retrieved_ids`` / ``relevant_ids`` to read other shapes.
"""

from __future__ import annotations

from typing import Any

from anatoolbox.analyze.calculate.base import PREFIX, STAGE
from anatoolbox.base import ToolContext
from anatoolbox.provenance import run_id_of
from anatoolbox.tool import BaseTool

TOOL_NAME = "calculate_retrieval_metrics"
DEFAULT_K = (1, 3, 5, 10)
MATCH_ON = ("source", "id")

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "description": (
                "One item per question: 'relevant' (ids that should be retrieved), 'retrieved' "
                "(a retrieval result, passages, or ids in rank order), optionally 'question' and 'id'."
            ),
            "items": {"type": "object"},
        },
        "k": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "default": list(DEFAULT_K),
            "description": "Cut-offs for precision@k, recall@k and hit@k.",
        },
        "match_on": {
            "type": "string",
            "enum": list(MATCH_ON),
            "default": "source",
            "description": "source = compare articles (source_id), counting each once; id = compare ids as given.",
        },
    },
    "required": ["results"],
}


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


class CalculateRetrievalMetricsTool(BaseTool):
    """Precision@k, recall@k, hit@k and reciprocal rank per question, and their means.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``metrics(retrieved, relevant, settings)`` — the measures for one question.
    * ``retrieved_ids(item, settings)`` / ``relevant_ids(item, settings)`` — read one item.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Compute retrieval metrics over a test set — precision@k, recall@k, hit@k and mean "
        "reciprocal rank — from what was retrieved and what should have been, per question "
        "and averaged."
    )
    input_schema = _INPUT_SCHEMA
    render_type = "table"

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments: the cut-offs ``k`` and how passages are matched."""
        k = args.get("k", list(DEFAULT_K))
        if isinstance(k, int) and not isinstance(k, bool):
            k = [k]
        if (
            not isinstance(k, list)
            or not k
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in k)
        ):
            raise self.input_error(
                f"k must be a positive integer or a list of them, got {k!r}.", argument="k", value=k
            )
        return {
            "k": sorted(set(k)),
            "match_on": self.choice_arg(args, "match_on", "source", MATCH_ON),
        }

    def retrieved_ids(self, item: dict[str, Any], settings: dict[str, Any]) -> list[str]:
        """Retrieved ids in rank order, each counted once."""
        value = item.get("retrieved")
        if isinstance(value, dict):
            value = value.get("passages")
        if not isinstance(value, list):
            raise self.input_error(
                "Each item's 'retrieved' must be a retrieval result, a list of passages, or a list of ids.",
                argument="results",
            )
        ids: list[str] = []
        for entry in value:
            if isinstance(entry, dict):
                key = entry.get("source_id") if settings["match_on"] == "source" else None
                entry = key or entry.get("id")
            if entry is not None and str(entry) not in ids:
                ids.append(str(entry))
        return ids

    def relevant_ids(self, item: dict[str, Any], settings: dict[str, Any]) -> set[str]:
        """The ids that should be retrieved for one item."""
        value = item.get("relevant")
        if isinstance(value, (str, int)):
            value = [value]
        if not isinstance(value, (list, tuple, set)):
            raise self.input_error(
                "Each item needs 'relevant': the ids that should be retrieved.", argument="results"
            )
        return {str(v) for v in value}

    def metrics(
        self, retrieved: list[str], relevant: set[str], settings: dict[str, Any]
    ) -> dict[str, float | None]:
        """Measures for one question. Values are floats, or None when undefined."""
        row: dict[str, float | None] = {}
        for k in settings["k"]:
            found = sum(1 for item in retrieved[:k] if item in relevant)
            row[f"precision@{k}"] = found / k
            row[f"recall@{k}"] = found / len(relevant) if relevant else None
            row[f"hit@{k}"] = 1.0 if found else 0.0
        first = next((rank for rank, item in enumerate(retrieved, 1) if item in relevant), None)
        row["reciprocal_rank"] = 1 / first if first else 0.0
        return row

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        items = args.get("results")
        if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
            raise self.input_error(
                "results must be a non-empty list of objects with 'relevant' and 'retrieved'.",
                code="missing_required_arguments",
                missing=["results"],
            )

        rows = []
        for position, item in enumerate(items, start=1):
            retrieved = self.retrieved_ids(item, settings)
            relevant = self.relevant_ids(item, settings)
            measures = self.metrics(retrieved, relevant, settings)
            rows.append(
                {
                    "id": item.get("id", position),
                    **({"question": item["question"]} if "question" in item else {}),
                    "retrieved": len(retrieved),
                    "relevant": len(relevant),
                    **{
                        name: (round(value, 4) if value is not None else None)
                        for name, value in measures.items()
                    },
                }
            )

        names = [n for n in rows[0] if n not in ("id", "question", "retrieved", "relevant")]
        summary = {name: _mean([row.get(name) for row in rows]) for name in names}
        if "reciprocal_rank" in summary:
            summary["mrr"] = summary.pop("reciprocal_rank")
        return {
            "questions": len(rows),
            "summary": summary,
            "per_question": rows,
            "provenance": self.provenance(
                {**settings, "questions": len(rows)},
                derived_from=[run_id_of(item.get("retrieved")) for item in items],
            ),
        }

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        rows = data["per_question"]
        return {
            "render_type": "table",
            "columns": list(rows[0]) if rows else [],
            "rows": rows,
            "caption": f"retrieval metrics over {data['questions']} questions",
        }
