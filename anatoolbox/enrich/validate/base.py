"""validate_ tool family — Enrich and interpret stage.

Check data, claims, analytical results, or conclusions against rules, independent evidence, expected invariants, or review criteria.

Naming: concrete tools are ``validate_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``validate_esg_claims``
- ``validate_benchmark_results``
- ``validate_ai_release_claims``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "validate_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = (
    "Artifact to validate; validation rules; reference evidence; tolerances; required checks."
)
ABSTRACT_OUTPUT: str = "Validation status, passed and failed checks, discrepancies, supporting evidence, and remediation actions."
TRANSFORMATION: str = "Check data, claims, analytical results, or conclusions against rules, independent evidence, expected invariants, or review criteria."


class ValidateTool(Protocol):
    """Structural contract for ``validate_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
