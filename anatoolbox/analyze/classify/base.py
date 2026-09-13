"""classify_ tool family — Analyze stage.

Assign one or more labels from a defined taxonomy or category system to an input object.

Naming: concrete tools are ``classify_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``classify_esrs_topics``
- ``classify_competitor_strategy``
- ``classify_ai_company_segment``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "classify_"
STAGE: str = "analyze"
STAGE_LABEL: str = "Analyze"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Structured object or evidence; taxonomy and version; labeling rules; multi-label and confidence policy."
ABSTRACT_OUTPUT: str = (
    "Classification labels, taxonomy identifiers, confidence, rationale, and supporting evidence."
)
TRANSFORMATION: str = (
    "Assign one or more labels from a defined taxonomy or category system to an input object."
)


class ClassifyTool(Protocol):
    """Structural contract for ``classify_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
