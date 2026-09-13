"""compare_ tool family — Analyze stage.

Evaluate similarities, differences, ranks, or changes between entities, cohorts, scenarios, or periods.

Naming: concrete tools are ``compare_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``compare_peer_esg_profiles``
- ``compare_distribution_networks``
- ``compare_ai_model_performance``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "compare_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Comparable datasets or objects; comparison dimensions; baseline or cohort; normalization policy."
ABSTRACT_OUTPUT: str = "Comparison results such as deltas, ranks, percentiles, similarities, or significance indicators."
TRANSFORMATION: str = "Evaluate similarities, differences, ranks, or changes between entities, cohorts, scenarios, or periods."


class CompareTool(Protocol):
    """Structural contract for ``compare_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
