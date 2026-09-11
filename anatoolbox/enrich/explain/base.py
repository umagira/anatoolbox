"""explain_ tool family — Enrich and interpret stage.

Produce a traceable, audience-appropriate rationale for a classification, metric, score, model result, or assessment.

Naming: concrete tools are ``explain_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``explain_materiality_assessment``
- ``explain_opportunity_score``
- ``explain_ai_landscape_classification``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "explain_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Result to explain; supporting and opposing evidence; method metadata; audience and detail level."
ABSTRACT_OUTPUT: str = "Structured or narrative explanation with cited evidence, component contributions, caveats, and assumptions."
TRANSFORMATION: str = "Produce a traceable, audience-appropriate rationale for a classification, metric, score, model result, or assessment."


class ExplainTool(Protocol):
    """Structural contract for ``explain_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
