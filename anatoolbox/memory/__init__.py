"""Memory for toolbox tools: session working slots plus cross-stage recordsets.

Two things live here, and they answer different questions.

``WorkingMemory``
    The original small slot bag on ``ToolContext.memory``. Pointers only,
    keyed by a fixed slot name. Still used where a tool needs one scalar
    handle (a document id) rather than a named dataset.

``Memory`` / ``Recordset``
    Cross-stage dataflow. The analytical workflow runs
    ``gather -> preprocess -> extract -> analyze -> enrich`` and stages skip
    freely, but a tool almost always builds on the output of an earlier one.
    A Recordset is that output, named, so the later tool can bind to it
    without the records ever passing through the model's context.

Lifecycle, end to end
---------------------

1. **Write.** A producing tool calls ``memory.remember(...)``, which allocates
   a handle (``passages_1``), records provenance (which tool, which args,
   which input handles) and stores either a reference (ids in a source system)
   or a value (records held here, for derived output with no source system).
   Handles never repeat, so an old log line can never point at new data.

2. **Advertise.** The agent loop calls ``memory.digest()`` and injects the
   result into the system prompt. The model sees handles, volumes and
   provenance — never records. A recordset absent from the digest is one the
   model will never use, so this step is what makes the store usable at all.
   The digest paginates rather than truncates.

3. **Bind.** A consuming tool calls ``memory.bind(object_type=...)``. An
   explicit ``input=`` handle wins; otherwise the newest recordset of that
   object_type is chosen, which reproduces the behaviour of the old
   single-slot pointer. The decision and the rejected candidates are logged.

4. **Read.** ``memory.records(recordset)`` materializes the records,
   re-fetching by id for reference mode. Resolvers are registered per
   reference store, so a new source is one registration.

5. **Compact.** At turn end the policy nominates stale recordsets. Anything
   another recordset derives from is tombstoned (metadata kept, payload
   dropped) instead of deleted, so lineage never dangles. Governance deletion
   is a separate ``purge`` that leaves nothing behind.

Every step above emits a plain-English log entry, and each turn ends with a
memory diff. Reading one session's log is the intended way to learn this.

Substrate and strategy are separate Protocols — ``MemoryStore`` for mechanism,
``MemoryPolicy`` for decisions — selected independently per project via
``AgentConfig.memory_store`` and ``AgentConfig.memory_policy``.
"""

from anatoolbox.memory.facade import Memory, register_ref_resolver
from anatoolbox.memory.oplog import (
    LEVEL_NORMAL,
    LEVEL_OFF,
    LEVEL_VERBOSE,
    ListSink,
    MemoryOpLog,
    MemoryOpSink,
    NullSink,
)
from anatoolbox.memory.policy import (
    DefaultMemoryPolicy,
    DigestPage,
    MemoryPolicy,
    RetainAllMemoryPolicy,
)
from anatoolbox.memory.recordset import (
    PROJECT_SCOPE,
    SESSION_SCOPE,
    Recordset,
    reference_ref,
    value_ref,
)
from anatoolbox.memory.store import (
    InMemoryRecordsetStore,
    JsonFileRecordsetStore,
    MemoryStore,
    NullRecordsetStore,
    owner_key,
)
from anatoolbox.memory.working_memory import WorkingMemory

__all__ = [
    "LEVEL_NORMAL",
    "LEVEL_OFF",
    "LEVEL_VERBOSE",
    "PROJECT_SCOPE",
    "SESSION_SCOPE",
    "DefaultMemoryPolicy",
    "DigestPage",
    "InMemoryRecordsetStore",
    "JsonFileRecordsetStore",
    "ListSink",
    "Memory",
    "MemoryOpLog",
    "MemoryOpSink",
    "MemoryPolicy",
    "MemoryStore",
    "NullRecordsetStore",
    "NullSink",
    "Recordset",
    "RetainAllMemoryPolicy",
    "WorkingMemory",
    "reference_ref",
    "owner_key",
    "register_ref_resolver",
    "value_ref",
]
