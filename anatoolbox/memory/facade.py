"""Memory — the one object tools and the agent loop actually touch.

Composes a ``MemoryStore`` (where bytes go) with a ``MemoryPolicy`` (what to
keep and show) and a ``MemoryOpLog`` (narration), so neither tools nor
``run_turn`` need to know which backend or strategy is configured.

Producer side::

    handle = context.recordsets.remember(
        object_type="passages",
        stage="gather",
        produced_by="retrieve_passages",
        args={"query": "agentic web"},
        ref=reference_ref(store="local", corpus="ai_media", ids=passage_ids),
        summary="47 passages matching 'agentic web'",
    ).handle

Consumer side::

    recordset = context.recordsets.bind(object_type="passages", requested=args.get("input"),
                                        tool_name="score_rag_answers")
    documents = context.recordsets.records(recordset, source_includes=[...])

``bind`` is where the ergonomics live. With no ``requested`` handle it picks
the newest recordset of the right object_type — which is exactly what the old
single-slot pointer did, so tools called the old way behave the same. With a
handle it binds explicitly, which the single-slot design could not express at
all. Either way the choice and its rejected alternatives are logged.
"""

from __future__ import annotations

from collections.abc import Callable

from anatoolbox.errors import ToolInputError
from anatoolbox.memory.oplog import MemoryOpLog, Timer
from anatoolbox.memory.policy import DigestPage, MemoryPolicy
from anatoolbox.memory.recordset import (
    REFERENCE_MODE,
    SESSION_SCOPE,
    VALUE_MODE,
    Recordset,
    payload_size_bytes,
)
from anatoolbox.memory.store import MemoryStore, owner_key
from anatoolbox.stages import STAGE_ORDER

NEWEST_COMPATIBLE = "newest_compatible"

# Stages whose output feeds a later stage rather than answering anything on its
# own. Derived from the canonical STAGE_ORDER so adding a stage upstream of
# "analyze" doesn't need a second edit here.
_INPUT_STAGES = frozenset(STAGE_ORDER[: STAGE_ORDER.index("analyze")])


def _is_input_stage(stage: str) -> bool:
    return stage in _INPUT_STAGES


# --- reference resolution ------------------------------------------------
#
# Reference-mode recordsets name a source system. Resolvers are registered by
# ``ref["store"]`` so a new source (another index, an object store, an API)
# becomes one registration rather than a change to this module.


def _resolve_inline(ref: dict, source_includes: list | None = None):
    records = list(ref.get("records") or [])
    return records, 0


_RESOLVERS: dict = {"inline": _resolve_inline}


def register_ref_resolver(store_name: str, resolver: Callable) -> None:
    """Teach Memory how to re-fetch records for a new reference store."""
    _RESOLVERS[store_name] = resolver


class Memory:
    """Store + policy + narration, bound to one session and project."""

    def __init__(
        self,
        store: MemoryStore,
        policy: MemoryPolicy,
        *,
        session_id: str,
        project: str,
        oplog: MemoryOpLog | None = None,
    ) -> None:
        self._store = store
        self._policy = policy
        self._session_id = session_id
        self._project = project
        self._log = oplog or MemoryOpLog()
        self._turn = 0
        self._created: list = []
        self._read: list = []
        self._evicted: list = []
        # Per dispatch batch (one loop iteration), not per turn: used to detect
        # a producer -> consumer chain that ran inside a single batch of tool
        # calls. See chained_in_step().
        self._step_created: list = []
        self._step_bound: list = []

    # --- identity ------------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def policy(self) -> MemoryPolicy:
        return self._policy

    @property
    def log(self) -> MemoryOpLog:
        return self._log

    def _owner(self, scope: str = SESSION_SCOPE) -> str:
        return owner_key(scope=scope, session_id=self._session_id, project=self._project)

    # --- write path ----------------------------------------------------

    def remember(
        self,
        *,
        object_type: str,
        stage: str,
        produced_by: str,
        ref: dict,
        args: dict | None = None,
        count: int | None = None,
        summary: str = "",
        derived_from=(),
        scope: str = SESSION_SCOPE,
    ) -> Recordset:
        """Store one tool call's output and return the named Recordset."""
        timer = Timer()
        owner = self._owner(scope)
        handle = self._store.allocate_handle(owner=owner, object_type=object_type)
        record = Recordset(
            handle=handle,
            object_type=object_type,
            stage=stage,
            produced_by=produced_by,
            session_id=self._session_id,
            project=self._project,
            ref=ref,
            args=dict(args or {}),
            derived_from=list(derived_from),
            scope=scope,
            count=count if count is not None else _infer_count(ref),
            summary=summary,
            size_bytes=payload_size_bytes(ref),
        )
        if not self._policy.admit(record):
            self._log.rejected(record, reason="policy.admit() returned False (nothing to store)")
            return record
        self._store.write(record)
        self._created.append(record.handle)
        self._step_created.append(record.handle)
        self._log.wrote(record, duration_ms=timer.ms)
        return record

    # --- read path -----------------------------------------------------

    def get(self, handle: str, *, scope: str = SESSION_SCOPE) -> Recordset | None:
        return self._store.read(handle, owner=self._owner(scope))

    def list(
        self,
        *,
        object_type: str | None = None,
        include_tombstoned: bool = False,
        limit: int | None = None,
        scope: str = SESSION_SCOPE,
    ) -> list:
        return self._store.query(
            owner=self._owner(scope),
            object_type=object_type,
            limit=limit,
            include_tombstoned=include_tombstoned,
        )

    def bind(
        self,
        *,
        object_type: str,
        tool_name: str,
        requested: str | None = None,
        scope: str = SESSION_SCOPE,
    ) -> Recordset:
        """Resolve the input recordset for a consuming tool.

        Explicit handle wins; otherwise the newest recordset of the right
        object_type. Raises ToolInputError naming the available handles when
        nothing fits, because a model that guessed a handle needs to see the
        real options rather than a generic failure.
        """
        timer = Timer()
        candidates = self.list(object_type=object_type, scope=scope)
        candidate_handles = [r.handle for r in candidates]

        if requested:
            found = self.get(requested, scope=scope)
            if found is None or found.tombstoned:
                available = [r.handle for r in self.list(scope=scope)]
                self._log.bind_failed(
                    object_type=object_type,
                    tool_name=tool_name,
                    available=available,
                    requested=requested,
                )
                raise ToolInputError(
                    code="unknown_recordset_handle",
                    message=(
                        "No stored data named '{requested}'. Available handles: {available}. "
                        "Pass one of these as input=, or call a retrieval tool first.".format(
                            requested=requested,
                            available=", ".join(available) if available else "(none)",
                        )
                    ),
                    tool_name=tool_name,
                    details={"requested": requested, "available": available},
                )
            self._step_bound.append(found.handle)
            self._log.bound(
                found,
                tool_name=tool_name,
                explicit=True,
                candidates=candidate_handles,
                rule="explicit",
                duration_ms=timer.ms,
            )
            return found

        if not candidates:
            available = [r.handle for r in self.list(scope=scope)]
            self._log.bind_failed(object_type=object_type, tool_name=tool_name, available=available)
            raise ToolInputError(
                code="missing_required_input",
                message=(
                    "No '{object_type}' data is stored in this session. Run a tool that "
                    "produces it first, or pass records explicitly. Available handles: "
                    "{available}.".format(
                        object_type=object_type,
                        available=", ".join(available) if available else "(none)",
                    )
                ),
                tool_name=tool_name,
                details={"object_type": object_type, "available": available},
            )

        chosen = candidates[0]
        self._step_bound.append(chosen.handle)
        self._log.bound(
            chosen,
            tool_name=tool_name,
            explicit=False,
            candidates=candidate_handles,
            rule=NEWEST_COMPATIBLE,
            duration_ms=timer.ms,
        )
        return chosen

    def records(self, record: Recordset, *, source_includes: list | None = None) -> list:
        """Materialize a recordset's records, re-fetching by id when needed."""
        timer = Timer()
        if record.tombstoned:
            raise ToolInputError(
                code="recordset_tombstoned",
                message=(
                    f"'{record.handle}' was evicted and only its metadata remains. Re-run "
                    f"{record.produced_by} to rebuild it."
                ),
                tool_name=record.produced_by,
                details={"handle": record.handle},
            )
        store_name = str(record.ref.get("store") or "inline")
        resolver = _RESOLVERS.get(store_name)
        if resolver is None:
            raise ToolInputError(
                code="unknown_reference_store",
                message=(
                    f"No resolver registered for reference store '{store_name}'. Register one with "
                    "anatoolbox.memory.register_ref_resolver()."
                ),
                tool_name=record.produced_by,
                details={"handle": record.handle, "store": store_name},
            )
        documents, missing = resolver(record.ref, source_includes)
        self._read.append(record.handle)
        self._log.read_records(
            record, resolved=len(documents), missing=missing, duration_ms=timer.ms
        )
        return documents

    def digest(self, *, page: int = 1, scope: str = SESSION_SCOPE) -> DigestPage:
        """One page of what the model may bind to, plus a log line saying so."""
        timer = Timer()
        rendered = self._policy.digest(self.list(scope=scope), page=page)
        self._log.digested(rendered, duration_ms=timer.ms)
        return rendered

    # --- lifecycle -----------------------------------------------------

    def dependents(self, handle: str, *, scope: str = SESSION_SCOPE) -> list:
        """Handles whose lineage names ``handle``. Drives tombstone-vs-delete."""
        return [
            r.handle
            for r in self.list(include_tombstoned=True, scope=scope)
            if handle in (r.derived_from or [])
        ]

    def compact(self, *, scope: str = SESSION_SCOPE) -> list:
        """Apply the policy's eviction decisions. Returns evicted handles."""
        evicted = []
        for record, reason in self._policy.evictable(self.list(scope=scope)):
            dependents = self.dependents(record.handle, scope=scope)
            tombstoned = bool(dependents)
            if self._store.delete(record.handle, owner=self._owner(scope), tombstone=tombstoned):
                evicted.append(record.handle)
                self._evicted.append(record.handle)
                self._log.evicted(
                    record, reason=reason, tombstoned=tombstoned, dependents=dependents
                )
        return evicted

    def purge(self, handle: str, *, scope: str = SESSION_SCOPE) -> list:
        """Governance deletion: remove entirely, cascading to derived copies.

        Distinct from eviction on purpose. Eviction is a cache decision and may
        tombstone; a purge must leave nothing behind, including anything
        promoted to project scope.
        """
        removed = []
        cascaded = self.dependents(handle, scope=scope)
        for target in [handle] + cascaded:
            if self._store.delete(target, owner=self._owner(scope), tombstone=False):
                removed.append(target)
        self._log.purged(removed, requested=handle, cascaded=cascaded)
        return removed

    def begin_turn(self, turn: int) -> None:
        self._turn = turn
        self._created = []
        self._read = []
        self._evicted = []
        self.begin_step()

    def begin_step(self) -> None:
        """Start a new dispatch batch. Call once per agent-loop iteration."""
        self._step_created = []
        self._step_bound = []

    def chained_in_step(self) -> bool:
        """Did a consumer in this batch bind to something produced in it?

        True means a real multi-stage chain executed inside one batch of tool
        calls — `retrieve_passages` then `score_rag_answers` on what it just
        fetched.
        """
        produced_here = set(self._step_created)
        return any(handle in produced_here for handle in self._step_bound)

    def unconsumed_input_in_step(self) -> list[Recordset]:
        """Raw upstream data this batch fetched that nothing has used yet.

        A `gather`/`extract` recordset is an *input* to later stages, not an
        answer in itself, so one sitting unconsumed at the end of a batch means
        the workflow is halfway done. `analyze` and later outputs are excluded:
        a mentions table is a result the user can read, so producing one is not
        evidence of unfinished work.
        """
        consumed = set(self._step_bound)
        pending = []
        for handle in self._step_created:
            if handle in consumed:
                continue
            record = self.get(handle)
            if record is not None and _is_input_stage(record.stage):
                pending.append(record)
        return pending

    def continuation_reason(self) -> str | None:
        """Why the loop should keep going despite an `end_turn_after` tool.

        None means nothing suggests unfinished work, so the early stop stands.
        The string is logged, because "the turn did not end where the tool
        schema said it would" is exactly the kind of silent control-flow
        decision that is miserable to debug after the fact.
        """
        if self.chained_in_step():
            return (
                "a tool in this batch consumed data produced in the same batch, so the model "
                "should get to report on the derived result"
            )
        pending = self.unconsumed_input_in_step()
        if pending:
            handles = ", ".join(record.handle for record in pending)
            return (
                f"{handles} holds freshly fetched data that no tool has used yet, so the model "
                "should get to continue the workflow it started"
            )
        return None

    def end_turn(self, *, scope: str = SESSION_SCOPE) -> None:
        """Emit the per-turn memory diff — the highest-signal log entry."""
        everything = self.list(include_tombstoned=True, scope=scope)
        live = [r for r in everything if not r.tombstoned]
        self._log.turn_summary(
            turn=self._turn,
            created=list(self._created),
            read=list(self._read),
            evicted=list(self._evicted),
            live_count=len(live),
            tombstone_count=len(everything) - len(live),
            total_bytes=sum(r.size_bytes for r in live),
        )


def _infer_count(ref: dict) -> int:
    if ref.get("mode") == REFERENCE_MODE:
        return len(ref.get("ids") or [])
    if ref.get("mode") == VALUE_MODE:
        return len(ref.get("records") or [])
    return 0


__all__ = ["NEWEST_COMPATIBLE", "Memory", "register_ref_resolver"]
