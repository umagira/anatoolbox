"""``classify_`` tools — Analyze stage.

Assign one or more labels from a defined taxonomy or category system to an input object.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``classify_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.classify.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ClassifyTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ClassifyTool",
]
