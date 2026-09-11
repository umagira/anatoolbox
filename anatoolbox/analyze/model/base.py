"""model_ tool family — Analyze stage.

Execute a statistical, forecasting, scenario, simulation, or optimization model.

Naming: concrete tools are ``model_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``model_transition_risk``
- ``model_demand``
- ``model_ai_market_growth``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "model_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Prepared analytical dataset; model specification; parameters; assumptions; scenarios or constraints."
ABSTRACT_OUTPUT: str = "Predictions, scenarios, optimized values, uncertainty, diagnostics, parameters, and model metadata."
TRANSFORMATION: str = (
    "Execute a statistical, forecasting, scenario, simulation, or optimization model."
)


class ModelTool(Protocol):
    """Structural contract for ``model_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
