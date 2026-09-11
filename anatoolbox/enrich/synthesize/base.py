"""synthesize_ tool family — Enrich and interpret stage.

Combine multiple findings and evidence streams into an integrated profile, conclusion, landscape, or decision view.

Naming: concrete tools are ``synthesize_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``synthesize_stewardship_profile``
- ``synthesize_competitor_benchmark``
- ``synthesize_ai_landscape``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "synthesize_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Findings, claims, assessments, and evidence; synthesis objective; prioritization and contradiction policy."
ABSTRACT_OUTPUT: str = "Integrated findings or claims with evidence links, confidence, contradictions, dependencies, and open questions."
TRANSFORMATION: str = "Combine multiple findings and evidence streams into an integrated profile, conclusion, landscape, or decision view."


class SynthesizeTool(Protocol):
    """Structural contract for ``synthesize_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
