"""``parse_`` tools — Extract information stage.

Convert a file or source format into machine-readable structural components without substantial domain interpretation.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``parse_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.extract.parse.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ParseTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ParseTool",
]
