"""``explain_`` tools — Enrich and interpret stage.

Produce a traceable, audience-appropriate rationale for a classification, metric, score, model result, or assessment.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``explain_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.explain.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ExplainTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ExplainTool",
]
