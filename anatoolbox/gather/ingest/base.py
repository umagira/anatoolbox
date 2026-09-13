"""ingest_ tool family — Gather stage.

Import a supplied or connected dataset, document corpus, or prior work product into a managed workspace for repeated downstream use.

Naming: concrete tools are ``ingest_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``ingest_existing_dma``
- ``ingest_schedule_data``
- ``ingest_ai_company_registry``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "ingest_"
STAGE: str = "gather"
STAGE_LABEL: str = "Gather"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Files, records, or connector references; source metadata; schema hints; load and update policy."
ABSTRACT_OUTPUT: str = "Registered corpus or dataset with validated structure, canonical identifiers, load status, and provenance."
TRANSFORMATION: str = "Import a supplied or connected dataset, document corpus, or prior work product into a managed workspace for repeated downstream use."


class IngestTool(Protocol):
    """Structural contract for ``ingest_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
