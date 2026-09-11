"""``retrieve_`` tools — Gather stage.



This package ships the prefix *contract* only. Instantiate it for your
own objects as ``retrieve_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.gather.retrieve.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    RetrieveTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "RetrieveTool",
]
