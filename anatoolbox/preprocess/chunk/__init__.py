"""``chunk_`` tools — Preprocess stage.

Split source records into smaller, self-contained units — passages, sections, or token windows — sized for a downstream model, keeping each unit linked to its source.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``chunk_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.preprocess.chunk.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ChunkTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ChunkTool",
]
