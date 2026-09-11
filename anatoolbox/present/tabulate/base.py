"""tabulate_ tool family — Present and visualize stage.

Arrange structured findings into a reviewable comparison, evidence, register, or summary table.

Naming: concrete tools are ``tabulate_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``tabulate_stewardship_actions``
- ``tabulate_route_benchmarks``
- ``tabulate_ai_company_profiles``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "tabulate_"
STAGE: str = "present"
STAGE_LABEL: str = "Present and visualize"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Structured records; desired rows and columns; audience; sorting, filtering, grouping, and display rules."
ABSTRACT_OUTPUT: str = "Formatted table specification or tabular artifact with values, labels, units, status, and evidence links."
TRANSFORMATION: str = "Arrange structured findings into a reviewable comparison, evidence, register, or summary table."


class TabulateTool(Protocol):
    """Structural contract for ``tabulate_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
