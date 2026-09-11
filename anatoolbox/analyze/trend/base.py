"""trend_ tool family — Analyze stage.

Analyze ordered observations over time to characterize direction, rate of change, persistence, or structural shifts.

Naming: concrete tools are ``trend_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``trend_controversy_frequency``
- ``trend_topic_salience``
- ``trend_ai_hiring_activity``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "trend_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Time-indexed observations; measure definitions; baseline; smoothing, restatement, and break-detection policy."
ABSTRACT_OUTPUT: str = "Trend metrics, change points, direction, persistence, anomalies, and supporting time-series data."
TRANSFORMATION: str = "Analyze ordered observations over time to characterize direction, rate of change, persistence, or structural shifts."


class TrendTool(Protocol):
    """Structural contract for ``trend_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
