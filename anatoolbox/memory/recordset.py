"""Recordset — one tool call's output, named so later stages can bind to it.

A Recordset is the unit of cross-stage memory. `retrieve_passages` produces
one; `score_rag_answers` consumes it and produces another. The model never
sees the records themselves, only the handle (``articles_1``) and a one-line
summary, so a 4,000-document result costs the same context as a 12-document
one.

Two storage modes, and the distinction is the whole point:

``reference``
    The records live in a source system and are re-fetched by id
    (``{"mode": "reference", "store": "es", "index": ..., "ids": [...]}``).
    This is the discipline the old ``articles_memory`` pointer enforced:
    bodies stay in Elasticsearch.

``value``
    The records are held in the memory store itself. Derived output
    (``score_rag_answers``, ``extract_*``, ``synthesize_*``) has no source
    system to re-fetch from, so pointer-only memory cannot express it at all.
    That gap is why stage chaining needed this module.

Recordsets are immutable. A second ``retrieve_passages`` call allocates
``articles_2`` rather than overwriting ``articles_1``, which is what makes
comparison and branching possible. Deletion tombstones (drops the payload,
keeps metadata) when other recordsets record it in ``derived_from``, so the
lineage graph never dangles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

SESSION_SCOPE = "session"
PROJECT_SCOPE = "project"

REFERENCE_MODE = "reference"
VALUE_MODE = "value"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def reference_ref(*, store: str, ids: list, **location: object) -> dict:
    """Reference mode: ids living in ``store``, re-fetched on read.

    ``location`` carries whatever that store needs to find them — an index
    name, a corpus name, a bucket. Register the matching resolver with
    ``anatoolbox.memory.register_ref_resolver``.
    """
    return {
        "mode": REFERENCE_MODE,
        "store": str(store),
        "ids": [str(i) for i in ids],
        **location,
    }


def value_ref(records: list) -> dict:
    """Value mode: records carried by the memory store itself."""
    return {"mode": VALUE_MODE, "store": "inline", "records": list(records)}


def payload_size_bytes(ref: dict) -> int:
    """Rough serialized size of value-mode records; 0 for reference mode.

    Logged so developers can see a store growing without having to log the
    payloads themselves.
    """
    if ref.get("mode") != VALUE_MODE:
        return 0
    try:
        return len(json.dumps(ref.get("records") or [], default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


@dataclass
class Recordset:
    handle: str
    object_type: str
    stage: str
    produced_by: str
    session_id: str
    project: str
    ref: dict
    args: dict = field(default_factory=dict)
    derived_from: list = field(default_factory=list)
    scope: str = SESSION_SCOPE
    count: int = 0
    summary: str = ""
    size_bytes: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    # Tombstones keep metadata, summary and lineage while dropping the
    # payload. Set when a recordset is evicted but something still derives
    # from it (see MemoryStore.delete).
    tombstoned: bool = False

    @property
    def mode(self) -> str:
        return str(self.ref.get("mode") or REFERENCE_MODE)

    @property
    def is_reference(self) -> bool:
        return self.mode == REFERENCE_MODE

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Recordset:
        """Tolerates unknown keys so a stored file survives a new field."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def tombstone(self) -> Recordset:
        """Return a copy with the payload dropped and metadata retained."""
        stripped = dict(self.ref)
        stripped.pop("records", None)
        return Recordset(
            handle=self.handle,
            object_type=self.object_type,
            stage=self.stage,
            produced_by=self.produced_by,
            session_id=self.session_id,
            project=self.project,
            ref=stripped,
            args=dict(self.args),
            derived_from=list(self.derived_from),
            scope=self.scope,
            count=self.count,
            summary=self.summary,
            size_bytes=0,
            created_at=self.created_at,
            tombstoned=True,
        )

    @property
    def ordinal(self) -> int:
        """Trailing number in the handle, for stable newest-first ordering.

        Needed because ``created_at`` timestamps can collide within a turn,
        and a lexicographic fallback would rank ``articles_9`` above
        ``articles_10``.
        """
        _, _, tail = self.handle.rpartition("_")
        try:
            return int(tail)
        except ValueError:
            return 0

    def origin(self) -> str:
        """Provenance phrase: producing tool, its args, and input handles."""
        origin = f"{self.produced_by}({_format_args(self.args)})"
        if self.derived_from:
            origin += " from " + ", ".join(self.derived_from)
        return origin

    def describe(self) -> str:
        """One digest line: handle, volume, provenance. Never records."""
        line = f"{self.handle} | {self.count} {self.object_type} | {self.origin()}"
        if self.tombstoned:
            line += " | payload dropped"
        return line


def _format_args(args: dict, *, max_len: int = 80) -> str:
    """Compact, readable arg rendering for digest lines and log sentences."""
    if not args:
        return ""
    parts = []
    for key in sorted(args):
        value = args[key]
        if isinstance(value, str):
            rendered = f'"{value}"'
        elif isinstance(value, (list, tuple)):
            rendered = json.dumps(list(value), default=str)
        elif isinstance(value, dict):
            rendered = "{...}"
        else:
            rendered = json.dumps(value, default=str)
        parts.append(f"{key}={rendered}")
    joined = ", ".join(parts)
    if len(joined) > max_len:
        joined = joined[: max_len - 3] + "..."
    return joined
