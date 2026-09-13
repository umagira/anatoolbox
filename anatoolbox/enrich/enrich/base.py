"""enrich_ tool family — Enrich and interpret stage.

Add contextual attributes or relationships from another dataset, knowledge source, or domain model.

Naming: concrete tools are ``enrich_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``enrich_company_sector_metadata``
- ``enrich_records_with_macro_context``
- ``enrich_ai_company_profiles``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "enrich_"
STAGE: str = "enrich"
STAGE_LABEL: str = "Enrich and interpret"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Target objects; enrichment attributes; reference source; join keys; precedence and freshness policy."
ABSTRACT_OUTPUT: str = (
    "Enriched objects with added fields, source references, match quality, and unresolved gaps."
)
TRANSFORMATION: str = "Add contextual attributes or relationships from another dataset, knowledge source, or domain model."


class EnrichTool(Protocol):
    """Structural contract for ``enrich_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
