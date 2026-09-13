"""deduplicate_ tool family — Preprocess stage.

Identify and consolidate duplicate or near-duplicate records while preserving corroboration and source lineage.

Naming: concrete tools are ``deduplicate_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``deduplicate_controversy_events``
- ``deduplicate_price_quotes``
- ``deduplicate_ai_news_events``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "deduplicate_"
STAGE: str = "preprocess"
STAGE_LABEL: str = "Preprocess"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Source artifacts, observations, or events; similarity features; duplicate definition; clustering policy."
ABSTRACT_OUTPUT: str = "Canonical records or duplicate clusters with member references, merge decisions, and provenance."
TRANSFORMATION: str = "Identify and consolidate duplicate or near-duplicate records while preserving corroboration and source lineage."


class DeduplicateTool(Protocol):
    """Structural contract for ``deduplicate_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
