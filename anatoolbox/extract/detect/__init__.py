"""``detect_`` tools — Extract information stage.

Determine whether a predefined event, signal, condition, or pattern is present in source data.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``detect_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.extract.detect.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    DetectTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "DetectTool",
]
