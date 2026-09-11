"""``summarize_`` tools — Present and visualize stage.

Produce a concise, audience-specific representation of selected findings while preserving material caveats and evidence.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``summarize_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.present.summarize.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    SummarizeTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "SummarizeTool",
]
