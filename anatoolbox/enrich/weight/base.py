"""weight_ tool family — Enrich and interpret stage.

Apply an explicit weighting policy to evidence, criteria, components, stakeholders, sources, or scenarios.

Naming: concrete tools are ``weight_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``weight_esg_evidence``
- ``weight_selection_criteria``
- ``weight_ai_signal_sources``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "weight_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = (
    "Items and component values; weight matrix; caps, floors, exclusions, and missing-data rules."
)
ABSTRACT_OUTPUT: str = "Weighted values or evidence set, contribution breakdown, sensitivity information, and policy version."
TRANSFORMATION: str = "Apply an explicit weighting policy to evidence, criteria, components, stakeholders, sources, or scenarios."


class WeightTool(Protocol):
    """Structural contract for ``weight_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
