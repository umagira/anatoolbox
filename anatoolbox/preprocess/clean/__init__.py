"""``clean_`` tools — Preprocess stage.

Remove material that is not part of a source's substance — boilerplate, markup, navigation, consent notices, scraping errors — without changing the meaning of what remains.

This package ships the prefix *contract* only. Instantiate it for your
own objects as ``clean_<object>`` and register with
``anatoolbox.registry.register_tool``.
"""

from anatoolbox.preprocess.clean.base import (
    ABSTRACT_INPUT,
    ABSTRACT_OUTPUT,
    PREFIX,
    STAGE,
    STAGE_LABEL,
    TRANSFORMATION,
    CleanTool,
)

__all__ = [
    "ABSTRACT_INPUT",
    "ABSTRACT_OUTPUT",
    "PREFIX",
    "STAGE",
    "STAGE_LABEL",
    "TRANSFORMATION",
    "CleanTool",
]
