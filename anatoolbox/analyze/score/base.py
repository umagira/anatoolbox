"""score_ tool family — Analyze stage.

Apply a defined rubric, weights, and thresholds to produce a numerical, ordinal, or categorical score.

Naming: concrete tools are ``score_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``score_stewardship_priority``
- ``score_opportunity_attractiveness``
- ``score_ai_company_momentum``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "score_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = (
    "Evidence or metrics; scoring rubric; weights; thresholds; missing-data and override policy."
)
ABSTRACT_OUTPUT: str = (
    "Overall and component scores, score category, rationale, rubric version, and review flags."
)
TRANSFORMATION: str = "Apply a defined rubric, weights, and thresholds to produce a numerical, ordinal, or categorical score."


class ScoreTool(Protocol):
    """Structural contract for ``score_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
