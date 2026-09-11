"""``trend_`` tools — Analyze stage.

Analyze ordered observations over time to characterize direction, rate of change, persistence, or structural shifts.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``trend_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.trend.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    TrendTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "TrendTool",
]
