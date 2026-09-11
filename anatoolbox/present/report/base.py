"""report_ tool family — Present and visualize stage.

Compose a complete deliverable from findings, evidence, analyses, tables, and visualizations.

Naming: concrete tools are ``report_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``report_stewardship_research``
- ``report_market_strategy``
- ``report_ai_landscape``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "report_"
STAGE: str = "present"
STAGE_LABEL: str = "Present and visualize"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Approved content artifacts; report template; audience; required sections; citation and governance requirements."
ABSTRACT_OUTPUT: str = "Versioned report artifact with narrative, visuals, citations, methodology, appendices, and review status."
TRANSFORMATION: str = (
    "Compose a complete deliverable from findings, evidence, analyses, tables, and visualizations."
)


class ReportTool(Protocol):
    """Structural contract for ``report_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
