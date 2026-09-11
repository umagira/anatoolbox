"""``score_`` tools — Analyze stage.

Apply a defined rubric, weights, and thresholds to produce a numerical, ordinal, or categorical score.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``score_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.score.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ScoreTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ScoreTool",
]
