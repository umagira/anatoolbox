"""calibrate_ tool family — Enrich and interpret stage.

Set or adjust thresholds, scales, parameters, or mappings using reference data, historical outcomes, or expert input.

Naming: concrete tools are ``calibrate_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``calibrate_materiality_thresholds``
- ``calibrate_score_thresholds``
- ``calibrate_ai_momentum_thresholds``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "calibrate_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Scores or model outputs; reference dataset; calibration objective; expert constraints; acceptance criteria."
ABSTRACT_OUTPUT: str = "Calibrated parameters or thresholds with diagnostics, rationale, validation results, and version."
TRANSFORMATION: str = "Set or adjust thresholds, scales, parameters, or mappings using reference data, historical outcomes, or expert input."


class CalibrateTool(Protocol):
    """Structural contract for ``calibrate_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
