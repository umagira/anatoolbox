"""assess_ tool family — Enrich and interpret stage.

Apply a domain framework to reach a qualitative or categorical judgment about significance, quality, risk, opportunity, or materiality.

Naming: concrete tools are ``assess_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``assess_topic_materiality``
- ``assess_market_opportunity``
- ``assess_ai_company_strategic_relevance``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "assess_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Evidence and analysis results; assessment framework; decision criteria; thresholds; contextual constraints."
ABSTRACT_OUTPUT: str = "Assessment outcome, dimensions, evidence-backed rationale, uncertainty, counterevidence, and review state."
TRANSFORMATION: str = "Apply a domain framework to reach a qualitative or categorical judgment about significance, quality, risk, opportunity, or materiality."


class AssessTool(Protocol):
    """Structural contract for ``assess_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
