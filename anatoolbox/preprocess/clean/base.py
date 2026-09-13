"""clean_ tool family — Preprocess stage.

Remove material that is not part of a source's substance — boilerplate, markup, navigation, consent notices, scraping errors — without changing the meaning of what remains.

Cleaning decides what every later stage sees. Text an embedding model reads first,
or a snippet shown to an LLM, is only as good as what cleaning left in place.

Naming: concrete tools are ``clean_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``clean_boilerplate``
- ``clean_html_markup``
- ``clean_scraping_errors``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "clean_"
STAGE: str = "preprocess"
STAGE_LABEL: str = "Preprocess"

ABSTRACT_INPUT: str = "Raw text records; the fields to clean; cleaning rules, patterns, or detectors — for example, text repeated across many records from one source."
ABSTRACT_OUTPUT: str = "Cleaned records with unchanged identities, plus per-record provenance of what was removed and why."
TRANSFORMATION: str = "Remove material that is not part of a source's substance — boilerplate, markup, navigation, consent notices, scraping errors — without changing the meaning of what remains."


class CleanTool(Protocol):
    """Structural contract for ``clean_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
