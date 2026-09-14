"""``rewrite_`` tools — Gather stage.

Rewrite a user's question into one or more retrieval-ready queries — clarified, expanded, or decomposed into sub-questions — without answering it.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``rewrite_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.gather.rewrite.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    RewriteTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "RewriteTool",
]
