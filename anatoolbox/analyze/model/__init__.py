"""``model_`` tools — Analyze stage.

Execute a statistical, forecasting, scenario, simulation, or optimization model.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``model_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.model.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ModelTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ModelTool",
]
