from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# Render types describe *how a result is meant to be displayed*. They are a
# host/frontend concern, never sent to the model — see `ToolSchema.render_type`.
#
# The package ships a small generic set. Domain render types belong to the
# application: register them at startup with `register_render_type`, rather
# than editing this module.

MARKDOWN_RENDER_TYPE = "markdown"

#: Render types every host understands.
CORE_RENDER_TYPES = frozenset({"markdown", "json", "table", "chart", "synthesis"})

#: Types whose payload is structured data rather than a prose blob.
_STRUCTURED: set[str] = {"json", "table", "chart", "synthesis"}
_REGISTERED: set[str] = set(CORE_RENDER_TYPES)


def register_render_type(render_type: str, *, structured: bool = True) -> str:
    """Make ``render_type`` valid for `build_render_payload` / `build_tool_result`.

    Hosts call this at startup for their own display kinds (``"usecases"``,
    ``"timeline"``, ``"network"``, …). Returns the name, so it can wrap a constant.
    """
    name = str(render_type).strip()
    if not name:
        raise ValueError("render_type must be a non-empty string.")
    _REGISTERED.add(name)
    if structured:
        _STRUCTURED.add(name)
    return name


def render_types() -> frozenset:
    """Every currently valid render type (core + host-registered)."""
    return frozenset(_REGISTERED)


def structured_render_types() -> frozenset:
    """Render types whose payload is structured data."""
    return frozenset(_STRUCTURED)


def _known(render_type: str) -> bool:
    return render_type in _REGISTERED


def build_render_payload(render_type: str, **fields: Any) -> dict[str, Any]:
    """Build the inner render object — always `{render_type, ...fields}` at one level.

    Presentation helper for ``execute`` / ``present`` — not the pipeline data API.
    Pipelines should call ``tool.run(...)`` and use the returned dict directly.
    """
    if not _known(render_type):
        raise ValueError(
            f"Unknown render_type: {render_type!r}. Known: {sorted(_REGISTERED)}. "
            f"Register host-specific types with register_render_type()."
        )
    return {"render_type": render_type, **fields}


def build_tool_result(
    *, render_type: str, render: dict[str, Any] | None = None, **render_fields: Any
) -> str:
    """Serialize structured tool data into agent-facing JSON with a ``render`` envelope.

    Used by ``execute`` / ``present`` so the SSE agent can emit a ``render`` event.
    Pipelines that only need data should call ``tool.run(...)`` instead.
    """
    payload = render if render is not None else build_render_payload(render_type, **render_fields)
    if payload.get("render_type") != render_type:
        raise ValueError(
            f"render_type mismatch: expected {render_type!r}, got {payload.get('render_type')!r}"
        )
    return json.dumps({"render": payload})


def build_markdown_result(content: str) -> str:
    """Presentation helper: wrap markdown ``content`` for the agent ``execute`` path."""
    return build_tool_result(render_type=MARKDOWN_RENDER_TYPE, content=content)


def format_tool_output_markdown(*, tool_name: str, result: str, is_error: bool) -> str:
    """Markdown body for a forced tool-only turn — plain JSON, errors, or help text."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = "error" if is_error else "ok"
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict) and "error" in parsed:
            error = parsed["error"]
            summary_line = error.get("message", "Tool execution failed.")
            raw_json = json.dumps(parsed, indent=2)
            return (
                f"**Tool called:** `{tool_name}` ({status})\n\n"
                f"**Timestamp:** {timestamp}\n\n"
                f"**Error:** {summary_line}\n\n"
                f"**Details:**\n\n```json\n{raw_json}\n```\n"
            )
        raw_json = json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, TypeError):
        raw_json = json.dumps(result)
    return (
        f"**Tool called:** `{tool_name}` ({status})\n\n"
        f"**Timestamp:** {timestamp}\n\n"
        f"**Output:**\n\n```json\n{raw_json}\n```\n"
    )


def format_tool_help_markdown(
    *, name: str, description: str, input_schema: dict[str, Any] | None = None
) -> str:
    lines = [f"## `{name}`", "", description, ""]
    if input_schema:
        lines.extend(
            ["**Parameters:**", "", f"```json\n{json.dumps(input_schema, indent=2)}\n```", ""]
        )
    example_required = input_schema.get("required", []) if input_schema else []
    if example_required:
        example = {key: "<value>" for key in example_required}
        if "keywords" in example:
            example["keywords"] = ["<keyword>"]
        lines.append(f"**Example:** `/{name} {json.dumps(example)}`")
    else:
        lines.append(f"**Example:** `/{name}`")
    return "\n".join(lines)


def format_tools_list_markdown(tools: list[Any]) -> str:
    lines = ["## Available tools", ""]
    for tool in tools:
        lines.append(f"- **`{tool.schema.name}`** — {tool.schema.description}")
    lines.extend(["", "Use `/toolname?` for parameter help."])
    return "\n".join(lines)


def resolve_forced_tool_render(*, tool_name: str, result: str, is_error: bool) -> dict[str, Any]:
    """Pick the single render payload for a forced tool-only turn."""
    if not is_error:
        existing = _extract_render_from_result(result)
        if existing and existing.get("render_type") in _STRUCTURED:
            return existing
        if existing and existing.get("render_type") == MARKDOWN_RENDER_TYPE:
            return existing
    content = format_tool_output_markdown(tool_name=tool_name, result=result, is_error=is_error)
    return build_render_payload(MARKDOWN_RENDER_TYPE, content=content)


def _extract_render_from_result(result: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    render = parsed.get("render") if isinstance(parsed, dict) else None
    return render if isinstance(render, dict) and "render_type" in render else None
