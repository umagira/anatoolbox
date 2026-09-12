"""anatoolbox — a task-type framework for research and analytics agents.

Two ideas carry the package:

1. **A grammar of analytical work.** Six stages (prepare, gather, extract,
   analyze, enrich, present) and 34 task-type prefixes, each a contract with
   abstract input/output and a one-sentence transformation. An agent reasons
   about *what kind of step comes next*, not which function to call.

2. **Recordset dataflow.** Evidence moves between stages by handle, carrying
   provenance and lineage, and never passes through the model's context
   window. See ``anatoolbox.memory``.

This package ships contracts, not a tool catalog. Instantiate a prefix for
your own objects and register it — see ``anatoolbox.registry``.
"""

from anatoolbox.base import Tool, ToolContext, ToolProgress, ToolSchema
from anatoolbox.registry import (
    REGISTRIES,
    TOOL_REGISTRY,
    ToolRegistrationError,
    register_tool,
    registered_names,
    resolve_tools,
    tools_by_stage,
    unregister_tool,
)
from anatoolbox.stages import (
    RESEARCH_PIPELINE_STAGES,
    STAGE_ORDER,
    prefix_to_stage,
    prefixes_for_stage,
    stage_for_tool_name,
)

__version__ = "0.1.0"

__all__ = [
    "REGISTRIES",
    "RESEARCH_PIPELINE_STAGES",
    "STAGE_ORDER",
    "TOOL_REGISTRY",
    "Tool",
    "ToolContext",
    "ToolProgress",
    "ToolRegistrationError",
    "ToolSchema",
    "__version__",
    "prefix_to_stage",
    "prefixes_for_stage",
    "register_tool",
    "registered_names",
    "resolve_tools",
    "stage_for_tool_name",
    "tools_by_stage",
    "unregister_tool",
]


def register_reference_tools(*, registry: str = "default", replace: bool = False) -> list[str]:
    """Register the reference instantiations that ship with the package.

    These are examples of how to instantiate a prefix, not a catalog you are
    expected to consume — see the README. Importing anatoolbox registers
    nothing on its own; call this when you want them.
    """
    from anatoolbox.gather.ingest.ingest_corpus import IngestCorpusTool
    from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool

    tools = [IngestCorpusTool(), RetrievePassagesTool()]
    return [register_tool(t, registry=registry, replace=replace).schema.name for t in tools]


__all__.append("register_reference_tools")
