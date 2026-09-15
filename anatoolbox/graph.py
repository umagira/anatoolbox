"""Knowledge graphs from Stage 1 — load the graph you built and look things up in it.

anatoolbox does not build knowledge graphs. Entity and relation extraction, the
schema and the graph analysis are Stage 1 work. This module is the socket for
their output, so a RAG system can use the graph next to text retrieval, with
every fact still traceable to the articles it came from.

A graph is a list of **facts**, one per edge:

=================  ============  =====================================================
field              required      meaning
=================  ============  =====================================================
``subject``        yes           entity name, e.g. "Nvidia"
``relation``       yes           edge label from your schema, e.g. "develops"
``object``         yes           entity name, e.g. "Blackwell"
``subject_type``   no            entity type from your schema, e.g. "Company"
``object_type``    no            entity type from your schema, e.g. "Chip"
``source_ids``     recommended   ids of the articles the fact was extracted from
``date``           no            when the fact was reported (ISO date)
=================  ============  =====================================================

Further fields — a co-mention ``weight``, a confidence — are kept as they are.
``source_ids`` may be a list, a list literal (``"['1', '2']"``), a single
``source_id``, or ids separated by ``;`` or ``|``.

Load a graph from an edge table (CSV, TSV, JSON, JSONL), from NetworkX
node-link JSON, from a NetworkX graph object, or from dicts::

    graph = KnowledgeGraph.from_file("track_graph_edges.csv", name="agentic_web")
    graph = KnowledgeGraph.from_networkx(G, name="agentic_web")   # edge attributes become fields

The lookups are deliberately plain — exact entity names, breadth-first
neighbourhoods — so they are a baseline to improve on, not a method: aliases,
fuzzy entity linking, path ranking and graph embeddings are yours to add.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anatoolbox.corpus import _read_delimited, _read_json, _read_jsonl, parse_list_string

REQUIRED_FIELDS = ("subject", "relation", "object")
DEFAULT_RELATION = "related_to"


def normalize_entity(name: Any) -> str:
    """The key an entity is looked up by: whitespace collapsed, case folded."""
    return re.sub(r"\s+", " ", str(name if name is not None else "")).strip().casefold()


def _source_ids(value: Any) -> list[str]:
    value = parse_list_string(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in re.split(r"[;|]", str(value)) if part.strip()]


def describe_fact(fact: dict[str, Any]) -> str:
    """One readable line for a fact, as shown to a language model.

    ``Nvidia [Company] --develops--> Blackwell [Chip] (2025-03-18)``
    """

    def entity(role: str) -> str:
        kind = fact.get(f"{role}_type")
        return f"{fact[role]} [{kind}]" if kind else str(fact[role])

    when = f" ({fact['date']})" if fact.get("date") else ""
    return f"{entity('subject')} --{fact['relation']}--> {entity('object')}{when}"


def fact_source_ids(facts: Iterable[dict[str, Any]]) -> list[str]:
    """Article ids behind ``facts``, each once, in order of first appearance."""
    ids: list[str] = []
    for fact in facts:
        for source_id in fact.get("source_ids") or []:
            if source_id not in ids:
                ids.append(source_id)
    return ids


@dataclass
class KnowledgeGraph:
    """Facts (edges) with an index from each entity to the facts it takes part in."""

    name: str
    facts: list[dict[str, Any]]
    #: Your Stage 1 schema or data dictionary, as loaded (dict or text). Kept for reference.
    schema: Any = None
    source: str = ""
    _by_entity: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _names: dict[str, str] = field(default_factory=dict, repr=False)
    _types: dict[str, str] = field(default_factory=dict, repr=False)
    _pattern: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the entity index. Call after changing ``facts`` in place."""
        self._by_entity, self._names, self._types, self._pattern = {}, {}, {}, None
        for index, fact in enumerate(self.facts):
            for role in ("subject", "object"):
                key = normalize_entity(fact[role])
                self._names.setdefault(key, str(fact[role]).strip())
                indices = self._by_entity.setdefault(key, [])
                if not indices or indices[-1] != index:
                    indices.append(index)
                if fact.get(f"{role}_type"):
                    self._types.setdefault(key, str(fact[f"{role}_type"]))

    # --- construction ----------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        name: str,
        schema: Any = None,
        source: str = "",
        subject_field: str = "subject",
        relation_field: str = "relation",
        object_field: str = "object",
    ) -> KnowledgeGraph:
        """Build a graph from one dict per fact. Field names can be mapped."""
        facts = []
        for position, record in enumerate(records):
            row = {k: (None if v == "" else parse_list_string(v)) for k, v in dict(record).items()}
            core = {
                "subject": row.pop(subject_field, None),
                "relation": row.pop(relation_field, None),
                "object": row.pop(object_field, None),
            }
            missing = [k for k, v in core.items() if v is None or not str(v).strip()]
            if missing:
                raise ValueError(
                    f"Fact {position} has no {missing}. Every fact needs subject, relation and "
                    f"object (map other column names with subject_field=, relation_field=, "
                    f"object_field=). Fields present: {sorted(record)}"
                )
            fact: dict[str, Any] = {k: str(v).strip() for k, v in core.items()}
            listed = row.pop("source_ids", None)
            single = row.pop("source_id", None)
            fact["source_ids"] = _source_ids(listed if listed is not None else single)
            fact.update(row)
            facts.append(fact)
        if not facts:
            raise ValueError("A knowledge graph needs at least one fact.")
        return cls(name=name, facts=facts, schema=schema, source=source)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        schema: Any = None,
        **fields: str,
    ) -> KnowledgeGraph:
        """Load an edge table (``.csv``, ``.tsv``, ``.json``, ``.jsonl``) or NetworkX node-link JSON."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such graph file: {path}")
        suffix = path.suffix.lower()
        name = name or path.stem
        if suffix in (".csv", ".tsv"):
            records = _read_delimited(path, delimiter="\t" if suffix == ".tsv" else ",", limit=None)
        elif suffix == ".jsonl":
            records = _read_jsonl(path, limit=None)
        elif suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(data, dict)
                and isinstance(data.get("nodes"), list)
                and (isinstance(data.get("links"), list) or isinstance(data.get("edges"), list))
            ):
                return cls.from_node_link(data, name=name, schema=schema, source=str(path))
            records = _read_json(path, limit=None)
        else:
            raise ValueError(
                f"Unsupported graph format {suffix!r}. Use an edge table (.csv, .tsv, .json, .jsonl) "
                "or NetworkX node-link JSON; for GraphML, load it with networkx and use from_networkx."
            )
        return cls.from_records(records, name=name, schema=schema, source=str(path), **fields)

    @classmethod
    def from_node_link(
        cls,
        data: dict[str, Any],
        *,
        name: str,
        schema: Any = None,
        source: str = "",
        relation_attr: str = "relation",
        type_attr: str = "type",
    ) -> KnowledgeGraph:
        """From NetworkX node-link data (``networkx.node_link_data(G)``)."""
        nodes = {str(n.get("id")): n for n in data.get("nodes") or [] if isinstance(n, dict)}
        facts = []
        for link in data.get("links") or data.get("edges") or []:
            attributes = {k: v for k, v in link.items() if k not in ("source", "target", "key")}
            facts.append(
                _edge_fact(
                    link.get("source"),
                    link.get("target"),
                    attributes,
                    nodes,
                    relation_attr,
                    type_attr,
                )
            )
        return cls.from_records(facts, name=name, schema=schema, source=source or "node-link")

    @classmethod
    def from_networkx(
        cls,
        graph: Any,
        *,
        name: str,
        schema: Any = None,
        relation_attr: str = "relation",
        type_attr: str = "type",
    ) -> KnowledgeGraph:
        """From a NetworkX graph: one fact per edge; edge attributes become fact fields.

        The edge attribute ``relation_attr`` names the relation (default
        ``related_to`` when absent, e.g. for co-mention graphs); the node attribute
        ``type_attr`` gives entity types.
        """
        nodes = {str(node): dict(data) for node, data in graph.nodes(data=True)}
        facts = [
            _edge_fact(u, v, dict(data), nodes, relation_attr, type_attr)
            for u, v, data in graph.edges(data=True)
        ]
        return cls.from_records(facts, name=name, schema=schema, source="networkx")

    # --- lookups -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.facts)

    @property
    def entities(self) -> list[str]:
        return sorted(self._names.values(), key=str.casefold)

    def has_entity(self, entity: str) -> bool:
        return normalize_entity(entity) in self._by_entity

    def entity_type(self, entity: str) -> str | None:
        return self._types.get(normalize_entity(entity))

    def facts_about(
        self, entity: str, *, relations: Iterable[str] | None = None
    ) -> list[dict[str, Any]]:
        """Facts in which ``entity`` is the subject or the object."""
        allowed = set(relations) if relations is not None else None
        return [
            self.facts[i]
            for i in self._by_entity.get(normalize_entity(entity), [])
            if allowed is None or self.facts[i]["relation"] in allowed
        ]

    def neighbors(self, entity: str, *, relations: Iterable[str] | None = None) -> list[str]:
        """Entities one edge away from ``entity``."""
        key = normalize_entity(entity)
        found: list[str] = []
        for fact in self.facts_about(entity, relations=relations):
            other = fact["object"] if normalize_entity(fact["subject"]) == key else fact["subject"]
            if normalize_entity(other) != key and other not in found:
                found.append(other)
        return found

    def subgraph(
        self,
        entities: Iterable[str],
        *,
        hops: int = 1,
        relations: Iterable[str] | None = None,
        max_facts: int | None = None,
    ) -> list[dict[str, Any]]:
        """Facts within ``hops`` edges of any of ``entities``, nearer facts first."""
        allowed = set(relations) if relations is not None else None
        frontier = [normalize_entity(e) for e in entities if normalize_entity(e) in self._by_entity]
        reached = set(frontier)
        taken: set[int] = set()
        collected: list[dict[str, Any]] = []
        for _ in range(max(hops, 0)):
            next_frontier: list[str] = []
            for key in frontier:
                for index in self._by_entity.get(key, []):
                    fact = self.facts[index]
                    if allowed is not None and fact["relation"] not in allowed:
                        continue
                    if index not in taken:
                        taken.add(index)
                        collected.append(fact)
                    for role in ("subject", "object"):
                        other = normalize_entity(fact[role])
                        if other not in reached:
                            reached.add(other)
                            next_frontier.append(other)
            frontier = next_frontier
            if not frontier:
                break
        return collected[:max_facts] if max_facts is not None else collected

    def find_entities(self, text: str) -> list[str]:
        """Graph entities named in ``text``, in order of appearance.

        Case-insensitive whole-name matches, preferring the longest name where
        names overlap ("GPT-4o" over "GPT-4"). Plain string matching: a baseline
        for entity linking, which misses aliases and abbreviations the graph
        does not list.
        """
        if self._pattern is None:
            names = sorted((key for key in self._names if key), key=len, reverse=True)
            self._pattern = (
                re.compile(
                    r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)", re.IGNORECASE
                )
                if names
                else False
            )
        if not self._pattern:
            return []
        found: list[str] = []
        for match in self._pattern.finditer(re.sub(r"\s+", " ", text or "")):
            display = self._names.get(normalize_entity(match.group(1)))
            if display and display not in found:
                found.append(display)
        return found

    def describe(self) -> dict[str, Any]:
        relations = Counter(fact["relation"] for fact in self.facts)
        types = Counter(self._types.values())
        return {
            "name": self.name,
            "facts": len(self.facts),
            "entities": len(self._names),
            "relations": dict(relations.most_common()),
            "entity_types": dict(types.most_common()),
            "facts_with_sources": sum(1 for fact in self.facts if fact.get("source_ids")),
            "dated_facts": sum(1 for fact in self.facts if fact.get("date")),
            "source": self.source,
        }


def _edge_fact(
    subject: Any,
    obj: Any,
    attributes: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    relation_attr: str,
    type_attr: str,
) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "subject": subject,
        "relation": attributes.pop(relation_attr, None) or DEFAULT_RELATION,
        "object": obj,
    }
    for role, node in (("subject", subject), ("object", obj)):
        kind = (nodes.get(str(node)) or {}).get(type_attr)
        if kind:
            fact[f"{role}_type"] = kind
    fact.update(attributes)
    return fact


# --- graph registry ---------------------------------------------------------

_GRAPHS: dict[str, KnowledgeGraph] = {}


def register_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Make ``graph`` available by name. Returns it, for chaining."""
    _GRAPHS[graph.name] = graph
    return graph


def get_graph(name: str) -> KnowledgeGraph:
    try:
        return _GRAPHS[name]
    except KeyError:
        raise KeyError(
            f"No knowledge graph named {name!r}. Registered: {sorted(_GRAPHS)}. "
            "Load one with ingest_knowledge_graph or KnowledgeGraph.from_file(...) + register_graph(...)."
        ) from None


def graph_names() -> list[str]:
    return sorted(_GRAPHS)


def clear_graphs() -> None:
    _GRAPHS.clear()
