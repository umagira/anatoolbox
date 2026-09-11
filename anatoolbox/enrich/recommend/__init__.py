"""``recommend_`` tools — Enrich and interpret stage.

Convert evidence, assessments, and constraints into proposed actions, priorities, or decisions.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``recommend_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.recommend.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    RecommendTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "RecommendTool",
]
