"""``assess_`` tools — Enrich and interpret stage.

Apply a domain framework to reach a qualitative or categorical judgment about significance, quality, risk, opportunity, or materiality.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``assess_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.assess.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    AssessTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "AssessTool",
]
