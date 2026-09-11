"""``calibrate_`` tools — Enrich and interpret stage.

Set or adjust thresholds, scales, parameters, or mappings using reference data, historical outcomes, or expert input.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``calibrate_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.calibrate.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    CalibrateTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "CalibrateTool",
]
