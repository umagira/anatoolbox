"""``resolve_`` tools — Extract information stage.

Determine whether different references represent the same canonical entity, asset, location, event, or concept.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``resolve_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.extract.resolve.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ResolveTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ResolveTool",
]
