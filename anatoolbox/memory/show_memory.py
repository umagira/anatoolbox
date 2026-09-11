"""show_memory — inspect what the session currently remembers, as JSON.

Reports both memory paths: the working-memory slots and the recordsets a later
stage can bind to. Recordsets are reported as metadata only (handle,
object_type, counts, provenance, lineage) — never records, which is the same
rule the prompt digest follows.

Useful when a chain misbehaves: it answers "what could the model have bound
to?" directly, rather than inferring it from the conversation.
"""

from __future__ import annotations

from typing import Any, ClassVar

from anatoolbox.base import ToolContext, ToolSchema
from anatoolbox.errors import ToolInputError
from anatoolbox.render import build_tool_result

TOOL_NAME = "show_memory"
RENDER_TYPE = "json"


def _describe_recordset(record: Any) -> dict[str, Any]:
    """Metadata only. Deliberately excludes ref['records']."""
    return {
        "handle": record.handle,
        "object_type": record.object_type,
        "stage": record.stage,
        "produced_by": record.produced_by,
        "args": record.args,
        "derived_from": record.derived_from,
        "count": record.count,
        "summary": record.summary,
        "mode": record.mode,
        "index": record.ref.get("index"),
        "size_bytes": record.size_bytes,
        "created_at": record.created_at,
        "tombstoned": record.tombstoned,
    }


def _recordset_report(context: ToolContext) -> dict[str, Any]:
    if context.recordsets is None:
        return {"available": False, "recordsets": []}
    records = context.recordsets.list(include_tombstoned=True)
    return {
        "available": True,
        "recordsets": [_describe_recordset(r) for r in records],
        "live_count": sum(1 for r in records if not r.tombstoned),
        "tombstone_count": sum(1 for r in records if r.tombstoned),
    }


class ShowMemoryTool:
    """Return working-memory pointers and stored recordsets for this session."""

    prefix: ClassVar[str] = "show_"
    stage: ClassVar[str] = "present"
    tool_name: ClassVar[str] = TOOL_NAME

    schema = ToolSchema(
        name=TOOL_NAME,
        description=(
            "Show what this session remembers: stored data handles (with counts and "
            "provenance) plus working-memory pointers. Metadata only — never document "
            "bodies. Pass optional slot= to return one named working-memory slot."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "slot": {
                    "type": "string",
                    "description": (
                        "Optional memory slot name (e.g. passages, entities). "
                        "Omit to return the full working-memory object."
                    ),
                },
            },
            "required": [],
        },
        render_type=RENDER_TYPE,
        end_turn_after=True,
    )

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        slot = args.get("slot")
        if slot is not None:
            if not isinstance(slot, str) or not slot.strip():
                raise ToolInputError(
                    code="invalid_argument_value",
                    message="Argument 'slot' must be a non-empty string when provided.",
                    tool_name=TOOL_NAME,
                    details={"argument": "slot"},
                )
            slot = slot.strip()

        stored = _recordset_report(context)

        if context.memory is None:
            data: Any = {} if slot is None else None
            return {
                "slot": slot,
                "data": data,
                "found": False,
                **stored,
            }

        if slot is None:
            # Full bag — shallow copy so callers cannot mutate the live store.
            data = {key: value for key, value in context.memory.data.items()}
            return {
                "slot": None,
                "data": data,
                "found": True,
                **stored,
            }

        pointer = context.memory.get(slot)
        return {
            "slot": slot,
            "data": pointer,
            "found": pointer is not None,
            **stored,
        }

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str:
        return build_tool_result(render_type=RENDER_TYPE, **self.run(args, context=context))
