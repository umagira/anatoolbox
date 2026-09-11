"""summarize_ tool family — Present and visualize stage.

Produce a concise, audience-specific representation of selected findings while preserving material caveats and evidence.

Naming: concrete tools are ``summarize_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``summarize_stewardship_findings``
- ``summarize_market_outlook``
- ``summarize_ai_landscape_changes``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "summarize_"
STAGE: str = "present"
STAGE_LABEL: str = "Present and visualize"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Findings and evidence; audience; communication objective; length, tone, priority, and citation requirements."
ABSTRACT_OUTPUT: str = "Structured or narrative summary with prioritized findings, caveats, evidence references, and open issues."
TRANSFORMATION: str = "Produce a concise, audience-specific representation of selected findings while preserving material caveats and evidence."


class SummarizeTool(Protocol):
    """Structural contract for ``summarize_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
