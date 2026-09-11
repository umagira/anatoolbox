"""``calculate_`` tools — Analyze stage.

Produce a deterministic derived value using an explicit formula or set of calculation rules.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``calculate_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.calculate.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    CalculateTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "CalculateTool",
]
