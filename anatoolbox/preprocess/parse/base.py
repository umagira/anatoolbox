"""parse_ tool family — Preprocess stage.

Convert a file or source format into machine-readable structural components without substantial domain interpretation.

Naming: concrete tools are ``parse_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``parse_sustainability_report``
- ``parse_timetable``
- ``parse_ai_research_paper``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "parse_"
STAGE: str = "preprocess"
STAGE_LABEL: str = "Preprocess"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Raw source artifact; file or content type; requested structural elements; parser configuration."
ABSTRACT_OUTPUT: str = (
    "Document structure, sections, passages, tables, fields, and source locators."
)
TRANSFORMATION: str = "Convert a file or source format into machine-readable structural components without substantial domain interpretation."


class ParseTool(Protocol):
    """Structural contract for ``parse_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
