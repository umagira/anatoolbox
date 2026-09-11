"""visualize_ tool family — Present and visualize stage.

Create a domain-specific visual model such as a matrix, map, network, landscape, journey, or relationship diagram.

Naming: concrete tools are ``visualize_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``visualize_materiality_matrix``
- ``visualize_entity_network_overlap``
- ``visualize_ai_landscape_map``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "visualize_"
STAGE: str = "present"
STAGE_LABEL: str = "Present and visualize"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Structured objects and relationships; visual grammar; analytical purpose; interaction and display constraints."
ABSTRACT_OUTPUT: str = "Visualization specification or rendered artifact with encodings, legends, interaction states, and accessibility text."
TRANSFORMATION: str = "Create a domain-specific visual model such as a matrix, map, network, landscape, journey, or relationship diagram."


class VisualizeTool(Protocol):
    """Structural contract for ``visualize_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
