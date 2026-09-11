"""Tool registry — how a host application selects tools.

This package ships **prefix contracts, not a tool catalog**. There is no
built-in list of tools to import: you instantiate a prefix for your own
objects and register it here.

    from anatoolbox.registry import register_tool, resolve_tools

    register_tool(RetrievePassagesTool())
    tool, = resolve_tools(["retrieve_passages"])

Registration validates two things: that the object satisfies the dual-use
``Tool`` contract (``schema`` + ``run`` + ``execute``), and that its name
starts with a known prefix. The second check is the point of the framework —
a tool whose name does not name a task type has no place in a stage.

Named registries let one process serve several projects with different tool
sets (``registry="esg"``); ``resolve_tools`` reads the default unless told
otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from anatoolbox.base import Tool
from anatoolbox.stages import prefix_to_stage, stage_for_tool_name

DEFAULT_REGISTRY = "default"
#: Entry-point group third-party packages use to publish tools.
PLUGIN_GROUP = "anatoolbox.tools"

_REGISTRIES: dict[str, dict[str, Tool]] = {DEFAULT_REGISTRY: {}}

#: Live view of the default registry. Mutated in place — never rebound, so
#: ``from anatoolbox import TOOL_REGISTRY`` stays correct after registration.
TOOL_REGISTRY: dict[str, Tool] = _REGISTRIES[DEFAULT_REGISTRY]
REGISTRIES = _REGISTRIES


class ToolRegistrationError(ValueError):
    """A tool failed the registration contract."""


def _validate(tool: Any, *, allow_unknown_prefix: bool) -> str:
    schema = getattr(tool, "schema", None)
    if schema is None:
        raise ToolRegistrationError(
            f"{type(tool).__name__} has no `schema`; a tool must expose a ToolSchema."
        )
    name = getattr(schema, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ToolRegistrationError(
            f"{type(tool).__name__}.schema.name must be a non-empty string."
        )
    if not isinstance(getattr(schema, "input_schema", None), dict):
        raise ToolRegistrationError(f"{name}: schema.input_schema must be a JSON Schema dict.")
    for method in ("run", "execute"):
        if not callable(getattr(tool, method, None)):
            raise ToolRegistrationError(
                f"{name}: missing `{method}`. Tools are dual-use — `run` returns data "
                f"for pipelines, `execute` returns a JSON string for the agent."
            )
    if not allow_unknown_prefix and stage_for_tool_name(name) is None:
        raise ToolRegistrationError(
            f"{name!r} does not start with a known task-type prefix. "
            f"Name it `<prefix><object>` using one of: {', '.join(sorted(prefix_to_stage()))}. "
            f"Pass allow_unknown_prefix=True to bypass."
        )
    return name


def register_tool(
    tool: Tool,
    *,
    registry: str = DEFAULT_REGISTRY,
    replace: bool = False,
    allow_unknown_prefix: bool = False,
) -> Tool:
    """Register ``tool`` under ``tool.schema.name``. Returns the tool.

    Raises ToolRegistrationError if the name is already taken and
    ``replace`` is False, so two packages cannot silently shadow each other.
    """
    name = _validate(tool, allow_unknown_prefix=allow_unknown_prefix)
    target = _REGISTRIES.setdefault(registry, {})
    if name in target and not replace:
        raise ToolRegistrationError(
            f"{name!r} is already registered in {registry!r}. Pass replace=True to override."
        )
    target[name] = tool
    return tool


def unregister_tool(name: str, *, registry: str = DEFAULT_REGISTRY) -> None:
    """Remove ``name``. No error if it was never registered."""
    _REGISTRIES.get(registry, {}).pop(name, None)


def resolve_tools(tool_names: Iterable[str], registry: str = DEFAULT_REGISTRY) -> list[Tool]:
    """Look up tools by name, preserving order."""
    try:
        selected = _REGISTRIES[registry]
    except KeyError as exc:
        raise KeyError(
            f"Unknown tool registry: {exc.args[0]!r}. Known: {sorted(_REGISTRIES)}"
        ) from exc
    resolved: list[Tool] = []
    for name in tool_names:
        try:
            resolved.append(selected[name])
        except KeyError as exc:
            raise KeyError(
                f"Unknown tool name: {exc.args[0]!r}. "
                f"Registered in {registry!r}: {sorted(selected)}"
            ) from exc
    return resolved


def registered_names(registry: str = DEFAULT_REGISTRY) -> list[str]:
    """Sorted tool names in ``registry``."""
    return sorted(_REGISTRIES.get(registry, {}))


def registry_names() -> list[str]:
    """Sorted names of all registries."""
    return sorted(_REGISTRIES)


def clear_registry(registry: str = DEFAULT_REGISTRY) -> None:
    """Empty a registry in place (keeps the TOOL_REGISTRY alias valid)."""
    _REGISTRIES.get(registry, {}).clear()


def tools_by_stage(registry: str = DEFAULT_REGISTRY) -> dict[str, list[str]]:
    """Group registered tool names by the stage their prefix implies."""
    grouped: dict[str, list[str]] = {}
    for name in registered_names(registry):
        grouped.setdefault(stage_for_tool_name(name) or "unknown", []).append(name)
    return grouped


def load_plugins(group: str = PLUGIN_GROUP, *, registry: str = DEFAULT_REGISTRY) -> list[str]:
    """Register tools published by installed packages via entry points.

    A distribution exposes tools with::

        [project.entry-points."anatoolbox.tools"]
        retrieve_passages = "mypkg.tools:RetrievePassagesTool"

    Each entry point may resolve to a Tool instance, a zero-arg class, or a
    callable returning either. Returns the names registered.
    """
    from importlib.metadata import entry_points

    registered: list[str] = []
    for ep in entry_points(group=group):
        obj = ep.load()
        tool = (
            obj()
            if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "schema"))
            else obj
        )
        registered.append(register_tool(tool, registry=registry, replace=True).schema.name)
    return registered
