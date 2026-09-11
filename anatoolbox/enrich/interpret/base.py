"""interpret_ tool family — Enrich and interpret stage.

Explain the significance, implications, possible causes, or strategic meaning of analytical results in context.

Naming: concrete tools are ``interpret_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``interpret_proxy_voting_trends``
- ``interpret_capacity_shift``
- ``interpret_ai_investment_signals``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "interpret_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Analysis results; domain and business context; stakeholder or decision perspective; alternative explanations."
ABSTRACT_OUTPUT: str = "Interpretive claims, implications, hypotheses, limitations, and evidence or analysis references."
TRANSFORMATION: str = "Explain the significance, implications, possible causes, or strategic meaning of analytical results in context."


class InterpretTool(Protocol):
    """Structural contract for ``interpret_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
