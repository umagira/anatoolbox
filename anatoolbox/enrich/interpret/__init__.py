"""``interpret_`` tools — Enrich and interpret stage.

Explain the significance, implications, possible causes, or strategic meaning of analytical results in context.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``interpret_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.interpret.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    InterpretTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "InterpretTool",
]
