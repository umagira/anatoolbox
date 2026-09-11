"""``enrich_`` tools — Enrich and interpret stage.

Add contextual attributes or relationships from another dataset, knowledge source, or domain model.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``enrich_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.enrich.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    EnrichTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "EnrichTool",
]
