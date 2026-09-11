"""resolve_ tool family — Extract information stage.

Determine whether different references represent the same canonical entity, asset, location, event, or concept.

Naming: concrete tools are ``resolve_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``resolve_issuer_entities``
- ``resolve_location_entities``
- ``resolve_ai_company_entities``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "resolve_"
STAGE: str = "extract"
STAGE_LABEL: str = "Extract information"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Candidate references; match attributes; authoritative identifiers; ambiguity and conflict policy."
ABSTRACT_OUTPUT: str = (
    "Canonical identifiers, match decisions, candidate matches, confidence, and review flags."
)
TRANSFORMATION: str = "Determine whether different references represent the same canonical entity, asset, location, event, or concept."


class ResolveTool(Protocol):
    """Structural contract for ``resolve_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
