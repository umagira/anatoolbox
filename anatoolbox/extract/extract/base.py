"""extract_ tool family — Extract information stage.

Identify explicit facts, entities, metrics, statements, events, or relationships contained in source material.

Naming: concrete tools are ``extract_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``extract_esg_topics``
- ``extract_reported_prices``
- ``extract_model_benchmarks``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "extract_"
STAGE: str = "extract"
STAGE_LABEL: str = "Extract information"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Parsed or raw content; extraction schema; target fields or object types; inclusion and exclusion rules."
ABSTRACT_OUTPUT: str = "Structured evidence units or observations with values, evidence locators, confidence, and provenance."
TRANSFORMATION: str = "Identify explicit facts, entities, metrics, statements, events, or relationships contained in source material."


class ExtractTool(Protocol):
    """Structural contract for ``extract_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
