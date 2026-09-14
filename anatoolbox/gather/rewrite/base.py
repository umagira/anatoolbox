"""rewrite_ tool family — Gather stage.

Rewrite a user's question into one or more retrieval-ready queries — clarified, expanded, or decomposed into sub-questions — without answering it.

People ask vague, broad, or compound questions, and a raw question is often a
poor search query. A broad question retrieves a little of everything; a compound
one retrieves evidence for only one of its parts. Rewriting clarifies the terms,
expands a broad question into targeted variants, or decomposes a compound
question into sub-questions that are each retrieved and then fused. It runs
before retrieval and never answers the question itself.

Naming: concrete tools are ``rewrite_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``rewrite_query_for_retrieval``
- ``rewrite_question_by_decomposition``
- ``rewrite_query_for_keyword_search``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "rewrite_"
STAGE: str = "gather"
STAGE_LABEL: str = "Gather"

# The contract: what an instantiation consumes, produces, and does.
ABSTRACT_INPUT: str = "A user question; optional research or conversation context; a rewriting strategy such as clarify, expand, or decompose; what the corpus covers, including its domain and time span."
ABSTRACT_OUTPUT: str = "One or more retrieval queries, each with its purpose, plus exact terms worth matching lexically, linked to the original question."
TRANSFORMATION: str = "Rewrite a user's question into one or more retrieval-ready queries — clarified, expanded, or decomposed into sub-questions — without answering it."


class RewriteTool(Protocol):
    """Structural contract for ``rewrite_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
