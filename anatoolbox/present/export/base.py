"""export_ tool family — Present and visualize stage.

Convert a completed data or presentation artifact into a specified delivery format without changing its substantive meaning.

Naming: concrete tools are ``export_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``export_esg_research_workbook``
- ``export_benchmark_pack``
- ``export_ai_landscape_dataset``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "export_"
STAGE: str = "present"
STAGE_LABEL: str = "Present and visualize"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = (
    "Completed artifact; target format; packaging, naming, metadata, and access requirements."
)
ABSTRACT_OUTPUT: str = (
    "Exported file or package with format metadata, checksum or version, and delivery status."
)
TRANSFORMATION: str = "Convert a completed data or presentation artifact into a specified delivery format without changing its substantive meaning."


class ExportTool(Protocol):
    """Structural contract for ``export_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
