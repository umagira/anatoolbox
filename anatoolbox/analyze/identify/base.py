"""identify_ tool family — Analyze stage.

Derive candidate gaps, anomalies, clusters, opportunities, risks, or other noteworthy analytical objects from available results.

Naming: concrete tools are ``identify_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``identify_esg_evidence_gaps``
- ``identify_coverage_gaps``
- ``identify_ai_landscape_shifts``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "identify_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Structured observations or analysis results; identification criteria; thresholds; prioritization policy."
ABSTRACT_OUTPUT: str = "Candidate findings with type, magnitude or relevance, evidence, confidence, and suggested next action."
TRANSFORMATION: str = "Derive candidate gaps, anomalies, clusters, opportunities, risks, or other noteworthy analytical objects from available results."


class IdentifyTool(Protocol):
    """Structural contract for ``identify_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
