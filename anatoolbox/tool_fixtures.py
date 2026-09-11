"""Shared helpers for fixture-driven tool runs (CLI + debug API)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anatoolbox.base import Tool, ToolContext
from anatoolbox.memory import (
    InMemoryRecordsetStore,
    Memory,
    RetainAllMemoryPolicy,
    WorkingMemory,
    value_ref,
)
from anatoolbox.registry import resolve_tools

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "tools"


def load_fixture(path: str | Path) -> dict[str, Any]:
    """Load a tool fixture JSON file.

    Expected shape::

        {
          "project": "demo",
          "tool": "synthesize_competitor_summary",
          "args": { ... },
          "gather": [  # optional — real retrievals, run before the tool
            {"tool": "retrieve_usecases", "args": {...}}
          ],
          "recordsets": [  # optional — canned records, no source system
            {
              "object_type": "usecases",
              "produced_by": "retrieve_usecases",
              "stage": "gather",
              "records": [ {...}, ... ]
            }
          ]
        }

    ``gather`` gives the tool real evidence (real ids, real URLs) and needs a
    configured Elasticsearch; ``recordsets`` replays canned evidence offline.
    Capture a ``gather`` run with ``capture_recordsets`` to turn one into the
    other.
    """
    fixture_path = Path(path).expanduser().resolve()
    if not fixture_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Fixture root must be a JSON object: {fixture_path}")
    return data


def resolve_fixture_path(name_or_path: str) -> Path:
    """Resolve a fixture path; bare names look under ``fixtures/tools/``."""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    under_pack = FIXTURES_DIR / name_or_path
    if under_pack.is_file():
        return under_pack.resolve()
    if not name_or_path.endswith(".json"):
        under_pack = FIXTURES_DIR / f"{name_or_path}.json"
        if under_pack.is_file():
            return under_pack.resolve()
    raise FileNotFoundError(
        f"Fixture not found: {name_or_path!r} (searched cwd and {FIXTURES_DIR})"
    )


def seed_recordsets(memory: Memory, entries: list[dict[str, Any]] | None) -> list[str]:
    """Remember fixture recordsets; return allocated handles in order."""
    if not entries:
        return []
    handles: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each recordsets entry must be an object")
        object_type = str(entry.get("object_type") or "").strip()
        if not object_type:
            raise ValueError("recordsets entry requires object_type")
        records = entry.get("records") or []
        if not isinstance(records, list):
            raise ValueError(f"records for {object_type!r} must be a list")
        produced_by = str(entry.get("produced_by") or "fixture").strip() or "fixture"
        stage = str(entry.get("stage") or "gather").strip() or "gather"
        summary = str(entry.get("summary") or f"fixture {object_type}")
        record = memory.remember(
            object_type=object_type,
            stage=stage,
            produced_by=produced_by,
            ref=value_ref(records),
            count=len(records),
            summary=summary,
            args=dict(entry.get("args") or {}),
        )
        handles.append(record.handle)
    return handles


def run_gather_steps(
    steps: list[dict[str, Any]] | None,
    *,
    context: ToolContext,
    registry: str = "default",
) -> list[dict[str, Any]]:
    """Execute a fixture's ``gather`` retrievals against the real source system.

    Each step runs through the tool's own ``execute`` (the agent path), so the
    recordsets it leaves behind are the same ones a workflow run would produce
    and the consuming tool can auto-bind them. Returns one log entry per step.
    """
    if not steps:
        return []
    if context.recordsets is None:
        raise ValueError("gather steps need recordset memory in the ToolContext")
    log: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("Each gather entry must be an object")
        name = str(step.get("tool") or "").strip()
        if not name:
            raise ValueError("gather entry requires tool")
        args = step.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"args for gather step {name!r} must be an object")
        tool = resolve_tools([name], registry=registry)[0]
        before = {record.handle for record in context.recordsets.list()}
        runner = getattr(tool, "execute", None) or getattr(tool, "run", None)
        if not callable(runner):
            raise ValueError(f"gather tool {name!r} has no execute()/run()")
        runner(dict(args), context=context)
        produced = [record for record in context.recordsets.list() if record.handle not in before]
        log.append(
            {
                "tool": name,
                "handles": [record.handle for record in produced],
                "counts": {record.handle: record.count for record in produced},
            }
        )
    return log


def capture_recordsets(memory: Memory) -> list[dict[str, Any]]:
    """Snapshot live recordsets as fixture ``recordsets`` entries.

    Materializes reference-mode records (re-fetching from the source system) so
    the captured fixture replays offline — the point of capturing is to iterate
    on a prompt against real evidence without re-querying.
    """
    entries: list[dict[str, Any]] = []
    for record in sorted(memory.list(), key=lambda r: (r.object_type, r.ordinal)):
        entries.append(
            {
                "object_type": record.object_type,
                "produced_by": record.produced_by,
                "stage": record.stage,
                "summary": record.summary,
                "args": dict(record.args),
                "records": memory.records(record),
            }
        )
    return entries


def build_tool_context(
    *,
    project: str,
    session_id: str = "fixture",
    recordset_entries: list[dict[str, Any]] | None = None,
) -> tuple[ToolContext, list[str]]:
    """Build a ToolContext with optional seeded recordset memory."""
    memory = Memory(
        InMemoryRecordsetStore(),
        RetainAllMemoryPolicy(),
        session_id=session_id,
        project=project,
    )
    handles = seed_recordsets(memory, recordset_entries)
    ctx = ToolContext(
        project=project,
        session_id=session_id,
        memory=WorkingMemory(),
        recordsets=memory,
    )
    return ctx, handles


def _summarize_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in resolved.items():
        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            summary[key] = {
                "count": len(value),
                "sample_keys": sorted({k for item in value[:3] for k in item})[:12],
            }
        elif isinstance(value, list):
            summary[key] = {"count": len(value), "values": value[:8]}
        else:
            summary[key] = value
    return summary


def dry_run_tool(tool: Tool, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
    """Resolve inputs and build prompts without calling the LLM (when supported)."""
    out: dict[str, Any] = {
        "tool": tool.schema.name,
        "project": context.project,
        "mode": "dry_run",
    }
    resolve = getattr(tool, "_resolve_args", None)
    if not callable(resolve):
        out["note"] = (
            "Tool has no _resolve_args; dry-run only reports schema name. "
            "Pass --no-dry-run to execute."
        )
        out["args"] = args
        return out

    resolved = resolve(args, context=context)
    out["resolved"] = _summarize_resolved(resolved)

    build_prompt = getattr(tool, "_build_user_prompt", None)
    if callable(build_prompt):
        user_prompt = build_prompt(resolved)
        system_prompt = getattr(tool, "system_prompt", None)
        out["system_prompt"] = system_prompt
        out["system_prompt_chars"] = len(system_prompt or "")
        out["user_prompt"] = user_prompt
        out["user_prompt_chars"] = len(user_prompt)
    return out
