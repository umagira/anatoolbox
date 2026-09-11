"""``weight_`` tools — Enrich and interpret stage.

Apply an explicit weighting policy to evidence, criteria, components, stakeholders, sources, or scenarios.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``weight_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.weight.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    WeightTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "WeightTool",
]
