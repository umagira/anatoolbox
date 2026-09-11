"""``aggregate_`` tools — Analyze stage.

Combine multiple records into summaries by entity, category, cohort, geography, or time period.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``aggregate_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.aggregate.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    AggregateTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "AggregateTool",
]
