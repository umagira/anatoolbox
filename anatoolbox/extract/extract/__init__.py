"""``extract_`` tools — Extract information stage.

Identify explicit facts, entities, metrics, statements, events, or relationships contained in source material.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``extract_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.extract.extract.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ExtractTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ExtractTool",
]
