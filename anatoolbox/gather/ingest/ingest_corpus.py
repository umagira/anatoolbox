"""ingest_corpus — register a local file as a queryable corpus.

The Gather-stage entry point for work that starts from a file rather than a
service: point it at a CSV/JSONL/Parquet export, and every later stage can
take the result as its ``input``.

    articles = ingest_corpus(path="ai_media.csv", text_field="content")

With recordset memory (agents), the corpus is also remembered in *reference
mode*: the corpus name and the row ids, never the rows. Bodies are re-read
from the corpus on demand, so a 200k-document corpus costs the same in memory
as a 200-document one.

**Variants.** Subclass ``IngestCorpusTool`` and override ``load`` to read
another source (a database, an API dump, a Stage 1 output) or ``prepare`` to
reshape the records before they are registered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anatoolbox.base import ToolContext
from anatoolbox.corpus import (
    DEFAULT_ID_FIELD,
    DEFAULT_TEXT_FIELD,
    LocalCorpus,
    local_ref,
    register_corpus,
)
from anatoolbox.gather.ingest.base import PREFIX, STAGE
from anatoolbox.tool import BaseTool

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
        "parse_lists": {
            "type": "boolean",
            "description": (
                "Convert cells holding a list literal ('[\"para\", ...]') into real lists. "
                "Default true."
            ),
            "default": True,
        },
        "limit": {
            "type": "integer",
            "description": "Read at most this many records. Useful while iterating.",
            "minimum": 1,
        },
    },
    "required": ["path"],
}


class IngestCorpusTool(BaseTool):
    """Load a local file into a named, queryable corpus.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``load(settings)`` — read the source into a ``LocalCorpus``.
    * ``prepare(corpus, settings)`` — reshape or filter records before registering.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Register a local CSV/JSONL/Parquet file as a corpus that later stages can "
        "query. Returns a handle; pass it to retrieve_passages, or omit it there to "
        "use the most recent corpus."
    )
    input_schema = _INPUT_SCHEMA

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments. Recorded in provenance, with the file name rather than its path."""
        return {
            "path": self.text_arg(args, "path", required=True),
            "name": self.text_arg(args, "name") or None,
            "id_field": self.text_arg(args, "id_field") or DEFAULT_ID_FIELD,
            "text_field": self.text_arg(args, "text_field") or DEFAULT_TEXT_FIELD,
            "limit": self.int_arg(args, "limit", None, minimum=1),
            "parse_lists": args.get("parse_lists", True) is not False,
        }

    def load(self, settings: dict[str, Any]) -> LocalCorpus:
        """Read the source. Default: a local file via ``LocalCorpus.from_file``."""
        return LocalCorpus.from_file(
            settings["path"],
            name=settings["name"],
            id_field=settings["id_field"],
            text_field=settings["text_field"],
            limit=settings["limit"],
            parse_lists=settings["parse_lists"],
        )

    def prepare(self, corpus: LocalCorpus, settings: dict[str, Any]) -> LocalCorpus:
        """Reshape the loaded corpus before it is registered. Default: unchanged.

        After changing ``corpus.records`` in place, call ``corpus.refresh()``.
        """
        return corpus

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        try:
            corpus = self.prepare(self.load(settings), settings)
        except (FileNotFoundError, ValueError) as exc:
            raise self.input_error(str(exc), argument="path", value=settings["path"]) from exc

        register_corpus(corpus)
        described = corpus.describe()
        described["corpus"] = corpus.name
        described["handle"] = self._remember(context, corpus)
        shared = {k: v for k, v in settings.items() if k != "path"}
        described["provenance"] = self.provenance(
            {
                # The file name, not its path: provenance is meant to be shared.
                "source_file": Path(settings["path"]).name,
                **shared,
                "name": corpus.name,
                "records": len(corpus),
            }
        )
        return described

    def _remember(self, context: ToolContext, corpus: LocalCorpus) -> str | None:
        """Record the corpus by reference. Nothing here holds a document body."""
        if context.recordsets is None:
            return None
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=self.stage,
            produced_by=self.tool_name,
            args={"corpus": corpus.name, "text_field": corpus.text_field},
            ref=local_ref(corpus=corpus.name, ids=corpus.ids),
            count=len(corpus),
            summary=f"{len(corpus)} records from {corpus.name}",
            derived_from=[],
        )
        return record.handle
