"""chunk_ tool family — Preprocess stage.

Split source records into smaller, self-contained units — passages, sections, or token windows — sized for a downstream model, keeping each unit linked to its source.

Chunking decides the unit of retrieval in RAG. Embedding models read a fixed
window — often only a few hundred tokens — so a whole article is embedded as little
more than its opening. Retrieving chunks instead lets the part of a document that
matches the query compete on its own, and hands a model the passage that matters
rather than the start of the document.

Chunk size trades context against precision: chunks that are too small lose the
surrounding meaning, chunks that are too large dilute the match and crowd the prompt.
Common strategies split on document structure (paragraphs, headings), on token or
word windows with overlap, or on semantic boundaries.

Naming: concrete tools are ``chunk_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``chunk_by_size``
- ``chunk_passages_by_token_window``
- ``chunk_reports_for_rag``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "chunk_"
STAGE: str = "preprocess"
STAGE_LABEL: str = "Preprocess"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "Source records with text; a chunking strategy (structural boundaries, token or word windows, semantic breaks); target size and overlap; the tokenizer or model the chunks are sized for."
ABSTRACT_OUTPUT: str = "Chunk records, each with a stable id, its text, its position within the source, and lineage to the source record, carrying the source's metadata such as title, date, and url."
TRANSFORMATION: str = "Split source records into smaller, self-contained units — passages, sections, or token windows — sized for a downstream model, keeping each unit linked to its source."


class ChunkTool(Protocol):
    """Structural contract for ``chunk_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
