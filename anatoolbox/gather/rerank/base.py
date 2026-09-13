"""rerank_ tool family — Gather stage.

Re-score a candidate set of retrieved records against the query using their original text, and re-order them by that score.

Why this is its own step: embedding retrieval ranks compressed vector representations.
Compression loses information, and each vector captures a generic, query-independent
meaning of its document, so the most relevant records do not reliably reach the top.
A reranker reads each candidate's original text together with the query at inference
time. That recovers what compression lost and judges relevance in the context of the
question. Rerankers are slow where precomputed embeddings are fast, so the effective
arrangement is to retrieve a generous candidate set first, then rerank it and keep
the best few.

Typical implementations: cross-encoder models, hosted rerank APIs such as Cohere
Rerank, or an LLM judging query-record relevance.

Naming: concrete tools are ``rerank_<object>[_by_<dimension>][_for_<purpose>]``.

Example names:
- ``rerank_passages``
- ``rerank_passages_by_cross_encoder``
- ``rerank_articles_for_temporal_questions``
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from anatoolbox.base import ToolContext, ToolSchema

PREFIX: str = "rerank_"
STAGE: str = "gather"
STAGE_LABEL: str = "Gather"

ABSTRACT_INPUT: str = "A query; a candidate set of retrieved records, typically a handle to an earlier retrieval, with access to their original text; a reranking model; how many records to keep."
ABSTRACT_OUTPUT: str = "The candidate records re-ordered, and usually truncated, by query-conditioned relevance scores, with lineage to the candidate set they came from."
TRANSFORMATION: str = "Re-score a candidate set of retrieved records against the query using their original text, and re-order them by that score."


class RerankTool(Protocol):
    """Structural contract for ``rerank_*`` tools.

    Implement ``run`` for pipelines/notebooks (structured data) and
    ``execute`` for the agent (JSON string, optional render envelope).
    """

    schema: ToolSchema
    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]: ...

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str: ...
