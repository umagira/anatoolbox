"""benchmark_ tool family — Enrich and interpret stage.

Place an entity or result in context using a peer group, standard, target, historical baseline, or reference distribution.

Naming: concrete tools are ``benchmark_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``benchmark_esg_metrics``
- ``benchmark_unit_costs``
- ``benchmark_ai_model_performance``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "benchmark_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = (
    "Target measures; benchmark population or standard; period; comparability and cohort rules."
)
ABSTRACT_OUTPUT: str = "Relative position, deltas, percentiles, benchmark ranges, comparability notes, and source metadata."
TRANSFORMATION: str = "Place an entity or result in context using a peer group, standard, target, historical baseline, or reference distribution."


class BenchmarkTool(Protocol):
    """Structural contract for ``benchmark_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
