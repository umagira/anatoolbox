"""normalize_ tool family — Extract information stage.

Standardize values, labels, units, currencies, dates, periods, or taxonomies into a common representation.

Naming: concrete tools are ``normalize_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``normalize_emissions_metrics``
- ``normalize_price_components``
- ``normalize_ai_benchmark_names``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "normalize_"
STAGE: str = "extract"
STAGE_LABEL: str = "Extract information"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = "Structured observations; source conventions; target conventions; conversion and mapping policy."
ABSTRACT_OUTPUT: str = (
    "Normalized observations plus transformation metadata and links to original values."
)
TRANSFORMATION: str = "Standardize values, labels, units, currencies, dates, periods, or taxonomies into a common representation."


class NormalizeTool(Protocol):
    """Structural contract for ``normalize_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
