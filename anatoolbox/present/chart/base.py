"""chart_ tool family — Present and visualize stage.

Create a conventional quantitative chart that communicates a comparison, trend, distribution, composition, or relationship.

Naming: concrete tools are ``chart_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``chart_esg_topic_scores``
- ``chart_price_trends``
- ``chart_ai_funding_by_segment``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "chart_"
STAGE: str = "present"
STAGE_LABEL: str = "Present and visualize"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Chart-ready dataset; analytical message; dimensions and measures; chart and formatting constraints."
ABSTRACT_OUTPUT: str = "Chart specification or rendered chart with labels, units, annotations, accessibility text, and data lineage."
TRANSFORMATION: str = "Create a conventional quantitative chart that communicates a comparison, trend, distribution, composition, or relationship."


class ChartTool(Protocol):
    """Structural contract for ``chart_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
