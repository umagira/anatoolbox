"""``plan_`` tools — Prepare stage.



This package ships the prefix *contract* only. Instantiate it for your
own objects as ``plan_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.prepare.plan.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    PlanTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "PlanTool",
]
