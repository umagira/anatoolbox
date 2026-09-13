"""aggregate_ tool family — Analyze stage.

Combine multiple records into summaries by entity, category, cohort, geography, or time period.

Naming: concrete tools are ``aggregate_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``aggregate_votes_by_topic``
- ``aggregate_mentions_by_period``
- ``aggregate_ai_investment_by_segment``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "aggregate_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Structured records; grouping dimensions; aggregation functions; weighting and missing-data policy."
ABSTRACT_OUTPUT: str = "Aggregated dataset at a defined grain with measures, counts, and lineage to contributing records."
TRANSFORMATION: str = "Combine multiple records into summaries by entity, category, cohort, geography, or time period."


class AggregateTool(Protocol):
    """Structural contract for ``aggregate_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
