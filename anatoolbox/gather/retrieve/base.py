"""retrieve_ tool family — Gather stage.

Fetch data from an external source for the current workflow run without materially
transforming its domain meaning.

Naming: concrete tools are ``retrieve_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:

- ``retrieve_esg_reports``
- ``retrieve_price_quotes``
- ``retrieve_ai_funding_news``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "retrieve_"
STAGE: str = "gather"
STAGE_LABEL: str = "Gather"

# Spec abstractions (analytical_toolbox_workflow_spec.xlsx).
ABSTRACT_INPUT: str = (
    "Retrieval scope; query or identifiers; source configuration; "
    "time, geography, entity, and access constraints."
)
ABSTRACT_OUTPUT: str = (
    "Raw source artifacts or records with source metadata, retrieval metadata, and provenance."
)
TRANSFORMATION: str = (
    "Fetch data from an external source for the current workflow run without "
    "materially transforming its domain meaning."
)


class RetrieveTool(Protocol):
    """Structural contract for ``retrieve_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
