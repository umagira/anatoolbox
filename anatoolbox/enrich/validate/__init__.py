"""``validate_`` tools — Enrich and interpret stage.

Check data, claims, analytical results, or conclusions against rules, independent evidence, expected invariants, or review criteria.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``validate_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.validate.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ValidateTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ValidateTool",
]
