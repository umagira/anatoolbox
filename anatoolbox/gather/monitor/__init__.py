"""``monitor_`` tools — Gather stage.

Check defined sources over time for new, changed, or removed information and emit meaningful change events.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``monitor_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.gather.monitor.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    MonitorTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "MonitorTool",
]
