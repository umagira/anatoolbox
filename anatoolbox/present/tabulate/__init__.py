"""``tabulate_`` tools — Present and visualize stage.

Arrange structured findings into a reviewable comparison, evidence, register, or summary table.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``tabulate_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.present.tabulate.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    TabulateTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "TabulateTool",
]
