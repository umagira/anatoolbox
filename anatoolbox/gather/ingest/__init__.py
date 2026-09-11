"""``ingest_`` tools — Gather stage.

Import a supplied or connected dataset, document corpus, or prior work product into a managed workspace for repeated downstream use.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``ingest_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.gather.ingest.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    IngestTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "IngestTool",
]
