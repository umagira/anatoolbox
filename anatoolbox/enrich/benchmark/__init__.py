"""``benchmark_`` tools — Enrich and interpret stage.

Place an entity or result in context using a peer group, standard, target, historical baseline, or reference distribution.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``benchmark_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.enrich.benchmark.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    BenchmarkTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "BenchmarkTool",
]
