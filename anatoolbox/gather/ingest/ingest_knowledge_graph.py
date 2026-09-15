"""ingest_knowledge_graph — load a Stage 1 knowledge graph for use in RAG.

The socket between Stage 1 and Stage 3. Point it at the graph you stored —
an edge table or NetworkX node-link JSON — and, optionally, at its schema or
data dictionary:

    graph = ingest_knowledge_graph(path="agentic_web_edges.csv", schema_path="schema.md")

The graph is registered by name, so later steps look it up with
``anatoolbox.graph.get_graph(graph["graph"])``. The result describes it —
facts, entities, relation and entity-type counts, and how many facts point back
at source articles — so gaps show up before they distort an evaluation. See
``anatoolbox.graph`` for the fact format.

For a graph that lives in memory as a NetworkX object, skip this tool:
``register_graph(KnowledgeGraph.from_networkx(G, name=...))``.

**Variants.** Subclass and override ``load`` for another storage format, or
``prepare`` to filter facts (e.g. to your track's relation types) or add
aliases before the graph is registered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anatoolbox.base import ToolContext
from anatoolbox.gather.ingest.base import PREFIX, STAGE
from anatoolbox.graph import KnowledgeGraph, register_graph
from anatoolbox.tool import BaseTool

TOOL_NAME = "ingest_knowledge_graph"

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Edge table (.csv, .tsv, .json, .jsonl) or NetworkX node-link JSON.",
        },
        "name": {
            "type": "string",
            "description": "Name to register the graph under (default: the file name stem).",
        },
        "schema_path": {
            "type": "string",
            "description": "Optional schema or data dictionary (.json, .md, .txt), kept with the graph.",
        },
        "subject_field": {"type": "string", "default": "subject"},
        "relation_field": {"type": "string", "default": "relation"},
        "object_field": {"type": "string", "default": "object"},
    },
    "required": ["path"],
}


class IngestKnowledgeGraphTool(BaseTool):
    """Load a knowledge graph from a file and register it by name.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``load(settings)`` — read the graph into a ``KnowledgeGraph``.
    * ``prepare(graph, settings)`` — filter or enrich facts before registering.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Load a knowledge graph built in an earlier stage — an edge table or NetworkX node-link "
        "JSON, with an optional schema — and register it for graph-aware retrieval. Describes "
        "its facts, entities, relations and how many facts cite source articles."
    )
    input_schema = _INPUT_SCHEMA

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments. Recorded in provenance, with file names rather than paths."""
        return {
            "path": self.text_arg(args, "path", required=True),
            "name": self.text_arg(args, "name") or None,
            "schema_path": self.text_arg(args, "schema_path") or None,
            "subject_field": self.text_arg(args, "subject_field") or "subject",
            "relation_field": self.text_arg(args, "relation_field") or "relation",
            "object_field": self.text_arg(args, "object_field") or "object",
        }

    def load(self, settings: dict[str, Any]) -> KnowledgeGraph:
        """Read the graph. Default: an edge table or node-link JSON via ``KnowledgeGraph.from_file``."""
        return KnowledgeGraph.from_file(
            settings["path"],
            name=settings["name"],
            schema=self.load_schema(settings["schema_path"]),
            subject_field=settings["subject_field"],
            relation_field=settings["relation_field"],
            object_field=settings["object_field"],
        )

    def prepare(self, graph: KnowledgeGraph, settings: dict[str, Any]) -> KnowledgeGraph:
        """Change the graph before it is registered. Default: unchanged.

        After changing ``graph.facts`` in place, call ``graph.refresh()``.
        """
        return graph

    def load_schema(self, path: str | None) -> Any:
        """The schema file's content: parsed JSON for ``.json``, text otherwise."""
        if not path:
            return None
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"No such schema file: {file}")
        text = file.read_text(encoding="utf-8")
        return json.loads(text) if file.suffix.lower() == ".json" else text

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        try:
            graph = self.prepare(self.load(settings), settings)
        except (FileNotFoundError, ValueError) as exc:
            raise self.input_error(str(exc), argument="path", value=settings["path"]) from exc
        register_graph(graph)
        described = graph.describe()
        described["graph"] = graph.name
        described["has_schema"] = graph.schema is not None
        described["source"] = Path(settings["path"]).name
        shared = {k: v for k, v in settings.items() if k not in ("path", "schema_path")}
        described["provenance"] = self.provenance(
            {
                # File names, not paths: provenance is meant to be shared.
                "source_file": Path(settings["path"]).name,
                "schema_file": Path(settings["schema_path"]).name
                if settings["schema_path"]
                else None,
                **shared,
                "name": graph.name,
                "facts": len(graph),
            }
        )
        return described
