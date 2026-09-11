"""plan_ tool family — Prepare stage.

Turn a research request into a scoped, reviewable research brief / plan.
Does not gather evidence or produce findings.

Naming: concrete tools are ``plan_<object>[_by_<dimension>][_for_<purpose>]``.

"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "plan_"
STAGE: str = "prepare"
STAGE_LABEL: str = "Prepare"

ABSTRACT_INPUT: str = (
    "User research request; optional intent/keyword hints; runtime tool registry slice."
)
ABSTRACT_OUTPUT: str = (
    "Resolved research brief: question, selected scope parameters, soft research plan."
)
TRANSFORMATION: str = (
    "Infer and resolve research scope, then produce a reviewable research brief and plan."
)


class PlanTool(Protocol):
    """Structural contract for ``plan_*`` tools."""

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
