"""``rerank_`` tools — Gather stage.

Re-score a candidate set of retrieved records against the query using their original text, and re-order them by that score.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``rerank_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.gather.rerank.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    RerankTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "RerankTool",
]
