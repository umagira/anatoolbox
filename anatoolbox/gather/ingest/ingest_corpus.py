"""ingest_corpus — register a local file as a queryable corpus.

The Gather-stage entry point for work that starts from a file rather than a
service: point it at a CSV/JSONL/Parquet export, and every later stage can
bind the result by handle.

    ingest_corpus(path="ai_media.csv", text_field="content")
    # -> corpus_1

The recordset it produces is *reference mode*: it stores the corpus name and
the row ids, never the rows. Bodies are re-read from the corpus on demand, so
a 200k-document corpus costs the same in memory as a 200-document one.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from anatoolbox.base import ToolContext, ToolSchema
from anatoolbox.corpus import (
    DEFAULT_ID_FIELD,
    DEFAULT_TEXT_FIELD,
    LocalCorpus,
    local_ref,
    register_corpus,
)
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.ingest.base import PREFIX, STAGE

TOOL_NAME = "ingest_corpus"
OBJECT_TYPE = "corpus"

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to a .csv, .tsv, .json, .jsonl or .parquet file.",
        },
        "name": {
            "type": "string",
            "description": "Name to register the corpus under (default: the filename stem).",
        },
        "text_field": {
            "type": "string",
            "description": f"Column holding the text to search (default: {DEFAULT_TEXT_FIELD}).",
            "default": DEFAULT_TEXT_FIELD,
        },
        "id_field": {
            "type": "string",
            "description": (
                f"Column holding a stable record id (default: {DEFAULT_ID_FIELD}). "
                "Ids are synthesized if the column is missing or blank."
            ),
            "default": DEFAULT_ID_FIELD,
        },
        "limit": {
            "type": "integer",
            "description": "Read at most this many records. Useful while iterating.",
            "minimum": 1,
        },
    },
    "required": ["path"],
}


class IngestCorpusTool:
    """Load a local file into a named, queryable corpus."""

    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE
    tool_name: ClassVar[str] = TOOL_NAME

    schema = ToolSchema(
        name=TOOL_NAME,
        description=(
            "Register a local CSV/JSONL/Parquet file as a corpus that later stages can "
            "query. Returns a handle; pass it to retrieve_passages, or omit it there to "
            "use the most recent corpus."
        ),
        input_schema=_INPUT_SCHEMA,
        render_type="json",
    )

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        path = args.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ToolInputError(
                code="missing_required_arguments",
                message="Argument 'path' is required.",
                tool_name=TOOL_NAME,
                details={"missing": ["path"]},
            )
        try:
            corpus = LocalCorpus.from_file(
                path.strip(),
                name=(args.get("name") or "").strip() or None,
                id_field=args.get("id_field") or DEFAULT_ID_FIELD,
                text_field=args.get("text_field") or DEFAULT_TEXT_FIELD,
                limit=args.get("limit"),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ToolInputError(
                code="invalid_argument_value",
                message=str(exc),
                tool_name=TOOL_NAME,
                details={"argument": "path", "value": path},
            ) from exc

        register_corpus(corpus)
        described = corpus.describe()
        described["handle"] = self._remember(context, corpus)
        return described

    def _remember(self, context: ToolContext, corpus: LocalCorpus) -> str | None:
        """Record the corpus by reference. Nothing here holds a document body."""
        if context.recordsets is None:
            return None
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=STAGE,
            produced_by=TOOL_NAME,
            args={"corpus": corpus.name, "text_field": corpus.text_field},
            ref=local_ref(corpus=corpus.name, ids=corpus.ids),
            count=len(corpus),
            summary=f"{len(corpus)} records from {corpus.name}",
            derived_from=[],
        )
        return record.handle

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str:
        data = self.run(args, context=context)
        return json.dumps({"render": {"render_type": "json", **data}, **data})
