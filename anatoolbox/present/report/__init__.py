"""``report_`` tools — Present and visualize stage.

Compose a complete deliverable from findings, evidence, analyses, tables, and visualizations.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``report_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.present.report.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    ReportTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "ReportTool",
]
