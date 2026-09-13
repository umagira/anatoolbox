"""recommend_ tool family — Enrich and interpret stage.

Convert evidence, assessments, and constraints into proposed actions, priorities, or decisions.

Naming: concrete tools are ``recommend_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``recommend_engagement_priorities``
- ``recommend_market_strategy``
- ``recommend_ai_companies_to_monitor``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "recommend_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Validated findings; decision objective; available actions; constraints; risk tolerance; evaluation criteria."
ABSTRACT_OUTPUT: str = "Ranked or structured recommendations with rationale, expected effects, dependencies, risks, and evidence."
TRANSFORMATION: str = "Convert evidence, assessments, and constraints into proposed actions, priorities, or decisions."


class RecommendTool(Protocol):
    """Structural contract for ``recommend_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
