"""monitor_ tool family — Gather stage.

Check defined sources over time for new, changed, or removed information and emit meaningful change events.

Naming: concrete tools are ``monitor_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``monitor_stewardship_controversies``
- ``monitor_price_changes``
- ``monitor_ai_model_releases``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "monitor_"
STAGE: str = "gather"
STAGE_LABEL: str = "Gather"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = (
    "Monitoring scope; source set; entities or topics; change definition; frequency; thresholds."
)
ABSTRACT_OUTPUT: str = "New or changed source artifacts plus structured change events, timestamps, and prior-state references."
TRANSFORMATION: str = "Check defined sources over time for new, changed, or removed information and emit meaningful change events."


class MonitorTool(Protocol):
    """Structural contract for ``monitor_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
