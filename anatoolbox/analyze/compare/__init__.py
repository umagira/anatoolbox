"""``compare_`` tools — Analyze stage.

Evaluate similarities, differences, ranks, or changes between entities, cohorts, scenarios, or periods.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``compare_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.compare.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    CompareTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "CompareTool",
]
