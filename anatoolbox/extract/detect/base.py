"""detect_ tool family — Extract information stage.

Determine whether a predefined event, signal, condition, or pattern is present in source data.

Naming: concrete tools are ``detect_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``detect_esg_controversies``
- ``detect_supply_disruptions``
- ``detect_ai_product_launches``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "detect_"
STAGE: str = "extract"
STAGE_LABEL: str = "Extract information"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = (
    "Source content or observations; detection definition; thresholds; contextual constraints."
)
ABSTRACT_OUTPUT: str = (
    "Detected event or signal records with type, evidence, confidence, and occurrence metadata."
)
TRANSFORMATION: str = (
    "Determine whether a predefined event, signal, condition, or pattern is present in source data."
)


class DetectTool(Protocol):
    """Structural contract for ``detect_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
