"""``normalize_`` tools — Extract information stage.

Standardize values, labels, units, currencies, dates, periods, or taxonomies into a common representation.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``normalize_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.extract.normalize.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    NormalizeTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "NormalizeTool",
]
