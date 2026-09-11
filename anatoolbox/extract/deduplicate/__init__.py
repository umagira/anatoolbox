"""``deduplicate_`` tools — Extract information stage.

Identify and consolidate duplicate or near-duplicate records while preserving corroboration and source lineage.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``deduplicate_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.extract.deduplicate.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    DeduplicateTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "DeduplicateTool",
]
