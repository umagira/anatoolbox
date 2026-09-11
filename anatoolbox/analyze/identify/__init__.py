"""``identify_`` tools — Analyze stage.

Derive candidate gaps, anomalies, clusters, opportunities, risks, or other noteworthy analytical objects from available results.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``identify_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.analyze.identify.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    IdentifyTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "IdentifyTool",
]
