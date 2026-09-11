"""``synthesize_`` tools — Enrich and interpret stage.

Combine multiple findings and evidence streams into an integrated profile, conclusion, landscape, or decision view.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``synthesize_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.synthesize.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    SynthesizeTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "SynthesizeTool",
]
