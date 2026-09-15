"""BaseTool — what every shipped tool shares, so a subclass changes one step and keeps the rest.

A shipped tool splits its work in two:

* a fixed ``run()`` that reads arguments, binds its input, records provenance
  and shapes the result — the plumbing every variant needs, and
* a few small **hook methods** — the step a variant changes: how a text is
  split, how a corpus is ranked, how a prompt is built.

To try a different method, subclass the tool, give it a new ``tool_name`` and
override one hook::

    class ChunkBySentence(ChunkBySizeTool):
        tool_name = "chunk_by_sentence"
        description = "Split records into chunks of whole sentences."

        def split(self, text, settings):
            ...  # return [{"text": ...}, ...]

The subclass inherits argument checking, pipeline chaining and provenance; its
results name ``chunk_by_sentence`` as the tool that made them, so a comparison
table can tell baseline and variant apart.

A hook that needs a new argument adds it in two places: ``input_schema`` (so
agents and readers see it) and ``settings()`` (which reads and checks it, and
whose output is recorded in provenance)::

        input_schema = with_properties(ChunkBySizeTool.input_schema, {
            "max_sentences": {"type": "integer", "minimum": 1},
        })

        def settings(self, args):
            return {**super().settings(args),
                    "max_sentences": self.int_arg(args, "max_sentences", 8, minimum=1)}
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any, ClassVar

from anatoolbox.base import ToolContext, ToolSchema
from anatoolbox.errors import ToolInputError
from anatoolbox.provenance import make_provenance


def with_properties(input_schema: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``input_schema`` with extra argument ``properties`` — for subclasses adding arguments."""
    extended = copy.deepcopy(input_schema)
    extended.setdefault("properties", {}).update(copy.deepcopy(properties))
    return extended


class BaseTool:
    """Schema, argument helpers, provenance and ``execute`` for a tool.

    Subclasses set ``tool_name``, ``prefix``, ``stage``, ``description`` and
    ``input_schema``, and implement ``run``. ``tool_name`` must start with the
    task-type prefix (``chunk_``, ``retrieve_``, ...) for the registry to accept it.
    """

    tool_name: ClassVar[str] = ""
    prefix: ClassVar[str] = ""
    stage: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_schema: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
    render_type: ClassVar[str | None] = "json"

    @property
    def schema(self) -> ToolSchema:
        """Built from the class attributes, so a subclass that renames itself needs nothing else."""
        return ToolSchema(
            name=self.tool_name,
            description=self.description,
            input_schema=self.input_schema,
            render_type=self.render_type,
        )

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        raise NotImplementedError

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str:
        """The agent path: ``run`` plus a render envelope from ``render``."""
        data = self.run(args, context=context)
        return json.dumps({"render": self.render(data), **data}, default=str)

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        """How an agent frontend should display a result. Default: as JSON."""
        return {"render_type": self.render_type or "json", **data}

    # --- provenance --------------------------------------------------------

    def provenance(
        self, settings: dict[str, Any], derived_from: Iterable[str | None] = ()
    ) -> dict[str, Any]:
        """Provenance naming this tool — the subclass's name when run from a subclass."""
        return make_provenance(self.tool_name, settings=settings, derived_from=derived_from)

    # --- arguments ---------------------------------------------------------

    def input_error(
        self, message: str, *, code: str = "invalid_argument_value", **details: Any
    ) -> ToolInputError:
        return ToolInputError(code=code, message=message, tool_name=self.tool_name, details=details)

    def text_arg(self, args: dict[str, Any], name: str, *, required: bool = False) -> str:
        """A stripped string argument; ``""`` when absent unless ``required``."""
        value = str(args.get(name) or "").strip()
        if required and not value:
            raise self.input_error(
                f"Argument {name!r} is required.", code="missing_required_arguments", missing=[name]
            )
        return value

    def int_arg(
        self, args: dict[str, Any], name: str, default: int | None, *, minimum: int = 1
    ) -> int | None:
        """An integer argument of at least ``minimum``; ``default`` when absent (may be None)."""
        value = args.get(name)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            requirement = "a positive integer" if minimum == 1 else f"an integer >= {minimum}"
            raise self.input_error(
                f"{name} must be {requirement}, got {value!r}.",
                argument=name,
                value=value,
            )
        return value

    def choice_arg(
        self, args: dict[str, Any], name: str, default: str, allowed: Iterable[str]
    ) -> str:
        """A string argument restricted to ``allowed``."""
        allowed = list(allowed)
        value = str(args.get(name) or default).strip()
        if value not in allowed:
            raise self.input_error(
                f"{name} must be one of {allowed}, got {value!r}.", argument=name, value=value
            )
        return value
