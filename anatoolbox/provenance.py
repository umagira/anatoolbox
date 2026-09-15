"""Portable provenance for tool results — lineage without a memory store.

Every ``run()`` result carries a ``provenance`` block::

    {
        "run_id": "retrieve_passages-5f3a9c1e",
        "tool": "retrieve_passages",
        "anatoolbox_version": "0.1.0",
        "created_at": "2026-09-15T08:12:03+00:00",
        "settings": {"query": "...", "strategy": "sparse", "size": 30, ...},
        "derived_from": ["chunk_by_size-1b2c3d4e"],
    }

``settings`` are the effective arguments, defaults included, so a result can be
reproduced without guessing what was left implicit. ``derived_from`` names what
the result was made from: the ``run_id`` of a result object passed in as input
(pipelines and notebooks), or a recordset handle such as ``passages_2`` (agents
with recordset memory).

It is plain JSON. Save it next to your results, and a later reader can tell
which corpus, chunking, queries, reranker and model produced each answer —
without the process that ran them.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from anatoolbox._version import __version__


def new_run_id(tool_name: str) -> str:
    return f"{tool_name}-{uuid.uuid4().hex[:8]}"


def make_provenance(
    tool_name: str,
    *,
    settings: dict[str, Any],
    derived_from: Iterable[str | None] = (),
) -> dict[str, Any]:
    """A provenance block for one tool run. Empty entries in ``derived_from`` are dropped."""
    return {
        "run_id": new_run_id(tool_name),
        "tool": tool_name,
        "anatoolbox_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": json.loads(json.dumps(settings, default=str)),
        "derived_from": [str(ref) for ref in derived_from if ref],
    }


def is_result(value: Any) -> bool:
    """Whether ``value`` is a tool result carrying provenance."""
    return isinstance(value, dict) and isinstance(value.get("provenance"), dict)


def run_id_of(value: Any) -> str | None:
    """The ``run_id`` of a tool result, or None for anything else."""
    return value["provenance"].get("run_id") if is_result(value) else None


def lineage(results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per result — run id, tool, what it derived from, settings — for a table."""
    rows = []
    for result in results:
        record = result["provenance"]
        rows.append(
            {
                "run_id": record["run_id"],
                "tool": record["tool"],
                "derived_from": record["derived_from"],
                "settings": record["settings"],
            }
        )
    return rows
