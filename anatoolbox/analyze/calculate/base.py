"""calculate_ tool family — Analyze stage.

Produce a deterministic derived value using an explicit formula or set of calculation rules.

Naming: concrete tools are ``calculate_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``calculate_carbon_intensity``
- ``calculate_mention_share``
- ``calculate_ai_funding_growth``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "calculate_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = (
    "Validated input measures; formula parameters; units; period and denominator conventions."
)
ABSTRACT_OUTPUT: str = "Calculated metric with units, formula or method reference, component inputs, and quality flags."
TRANSFORMATION: str = (
    "Produce a deterministic derived value using an explicit formula or set of calculation rules."
)


class CalculateTool(Protocol):
    """Structural contract for ``calculate_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
