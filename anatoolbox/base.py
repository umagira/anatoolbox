from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from anatoolbox.memory.working_memory import WorkingMemory

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    # anatoolbox.memory.facade imports anatoolbox.errors, which is fine, but the
    # facade also needs ToolContext-adjacent types; keep this annotation-only.
    from anatoolbox.memory.facade import Memory


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[
        str, Any
    ]  # JSON Schema; each provider reshapes this to its own tool-def format
    # Static declaration of what this tool's result CAN render as, for docs/frontend-prep purposes only —
    # never sent to the model (providers pick name/description/input_schema explicitly, see
    # anthropic_provider._to_anthropic_tools / openai_provider's equivalent). Independent of whether a given
    # call's actual result embeds a "render" object (see agent_loop._extract_render) — that's the real,
    # per-result signal; this is just a hint for docs/sse_event_reference generation and the humans reading it.
    render_type: str | None = None
    # Slash commands only: if True, the slash path's `done` stop_reason is
    # `awaiting_user` (the UI should treat the render as the answer). The agent
    # loop ignores this flag — it never stops because a tool ran. Trailing
    # model prose after a `render` event is dropped instead. Never sent to the
    # model.
    end_turn_after: bool = False


@dataclass
class ToolContext:
    #: Routes project-scoped resources such as per-project API keys. Pipelines and
    #: notebooks can leave the default.
    project: str = "default"
    #: Correlates log lines within one conversation. Pipelines can leave it empty.
    session_id: str = ""
    # Session-scoped working memory (agent wraps Session.working_state). Holds
    # small pointers/handles only — never large document bodies. Pipelines and
    # notebooks pass WorkingMemory() or None.
    memory: WorkingMemory | None = None
    # Cross-stage recordset memory: named, provenance-tagged tool outputs a
    # later stage can bind to (`articles_1` -> `mentions_1`). This is the
    # "store" half of the context/state/store split — durable and queryable,
    # where `memory` above is per-run mutable state. A separate field rather
    # than a retype of `memory` so existing pointer-based tools keep working
    # unchanged. None in pipelines that don't need cross-stage chaining; tools
    # must degrade gracefully (accept explicit records) when it is absent.
    recordsets: Memory | None = None


class Tool(Protocol):
    """Dual-use tool contract.

    - ``run`` — core work; returns structured data for pipelines and notebooks.
    - ``execute`` — agent/SSE entry point; calls ``run`` (or equivalent), then
      wraps presentation (optional ``{"render": {...}}`` envelope) and returns a
      JSON string. Composite tools may make ``execute`` a generator that yields
      ``ToolProgress`` and returns a final string (PEP 380).
    """

    schema: ToolSchema

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...


@dataclass
class ToolProgress:
    """Yielded by a streaming/composite tool's ``execute()`` while the outer
    call is still running (see agent_loop._run_composite).

    Shapes:
    - ``text`` set — character/token delta for the outer tool's answer; agent_loop
      emits an SSE ``token`` event tagged with the outer ``tool_call_id``.
    - ``args`` set and ``result`` None — nested sub-call starting → ``tool_call``.
    - ``result`` set — nested sub-call finished → ``tool_result`` (+ optional render).

    A tool opts in by making ``execute`` a generator: ``yield`` ToolProgress,
    then ``return`` the final result string (PEP 380). Pipelines that only need
    data should call ``run`` instead (non-streaming).
    """

    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] | None = None
    result: str | None = None
    is_error: bool = False
    duration_ms: int | None = None
    text: str | None = None
