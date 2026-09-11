"""Narrated memory logging — how a developer learns this subsystem.

Memory failures are silent. A wrong auto-binding does not raise; it produces a
slightly worse answer. So the log is not decoration here, it is the primary
diagnostic surface, and every entry carries a plain-English sentence next to
its structured fields. Reading one session's log end to end should teach the
whole lifecycle without opening this package.

The sentence explains the *decision*, not just the fact. A bind records which
rule fired and which candidates lost; an eviction records which threshold was
crossed and why it tombstoned instead of deleting. Those two are where the
silent failures live.

Three rules keep "detailed" from becoming "unmanageable":

1. **Never log payloads.** Counts, byte sizes, truncated id lists. Recordsets
   exist to keep records out of the context window; the log must not smuggle
   them back in.
2. **Two levels, not five.** ``normal`` logs decisions (write, bind, evict,
   purge, turn summary). ``verbose`` adds traffic (read, digest). No logging
   framework, no per-module configuration.
3. **Best effort, always.** A sink failure is swallowed. Logging must never
   break a memory operation, exactly as ``agent_loop._log`` must never break a
   conversation turn.

The agent package adapts its ``ConversationLogger`` to ``MemoryOpSink`` so
memory entries land in the same per-session JSONL as ``tool_call`` and
``tool_result`` — one file, one timeline.
"""

from __future__ import annotations

import time
from typing import Protocol

from anatoolbox.memory.recordset import Recordset

LEVEL_NORMAL = "normal"
LEVEL_VERBOSE = "verbose"
LEVEL_OFF = "off"

# Ops that describe a decision. Always logged unless logging is off.
_DECISION_OPS = frozenset({"write", "bind", "bind_failed", "evict", "purge", "turn_summary"})
# Ops that describe traffic. Logged at verbose only.
_TRAFFIC_OPS = frozenset({"read", "digest"})

_MAX_LOGGED_IDS = 5


class MemoryOpSink(Protocol):
    """Destination for memory log entries."""

    def emit(self, entry: dict) -> None: ...


class NullSink:
    """Discards everything. Default when no sink is wired."""

    def emit(self, entry: dict) -> None:
        return None


class ListSink:
    """Collects entries in a list. For tests and notebooks."""

    def __init__(self) -> None:
        self.entries: list = []

    def emit(self, entry: dict) -> None:
        self.entries.append(entry)

    def ops(self) -> list:
        return [e.get("op") for e in self.entries]

    def explanations(self) -> list:
        return [e.get("explanation") for e in self.entries]


class Timer:
    """Wall-clock helper so every op can report duration_ms."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    @property
    def ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)


def human_bytes(size: int) -> str:
    """Readable size. Integer KB alone would report 666 bytes as '0 KB'."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _truncated_ids(ids: list) -> str:
    ids = [str(i) for i in ids]
    if len(ids) <= _MAX_LOGGED_IDS:
        return ", ".join(ids)
    head = ", ".join(ids[:_MAX_LOGGED_IDS])
    return f"{head}, ... (+{len(ids) - _MAX_LOGGED_IDS} more)"


def _origin(record: Recordset) -> str:
    return record.origin()


class MemoryOpLog:
    """Builds and emits narrated memory entries. One line per call site."""

    def __init__(self, sink: MemoryOpSink | None = None, *, level: str = LEVEL_NORMAL) -> None:
        self._sink = sink or NullSink()
        self.level = level

    def _enabled(self, op: str) -> bool:
        if self.level == LEVEL_OFF:
            return False
        if op in _TRAFFIC_OPS:
            return self.level == LEVEL_VERBOSE
        return True

    def _emit(
        self,
        op: str,
        *,
        explanation: str,
        handle: str | None = None,
        object_type: str | None = None,
        details: dict | None = None,
        duration_ms: int | None = None,
    ) -> None:
        if not self._enabled(op):
            return
        entry = {
            "op": op,
            "handle": handle,
            "object_type": object_type,
            "explanation": explanation,
            "details": details or {},
            "duration_ms": duration_ms,
        }
        try:
            self._sink.emit(entry)
        except Exception:  # noqa: BLE001 - logging must never break memory
            pass

    # --- decisions -----------------------------------------------------

    def wrote(
        self,
        record: Recordset,
        *,
        duration_ms: int | None = None,
        evicted: list | None = None,
    ) -> None:
        if record.is_reference:
            where = "by reference to {} index '{}' ({} ids: {})".format(
                record.ref.get("store", "?"),
                record.ref.get("index", "?"),
                len(record.ref.get("ids") or []),
                _truncated_ids(record.ref.get("ids") or []),
            )
        else:
            where = f"by value in the memory store ({record.size_bytes} bytes; no source system to re-fetch from)"
        explanation = f"Stored {record.count} {record.object_type} as {record.handle} {where}, produced by {_origin(record)}."
        if evicted:
            explanation += " Same write evicted: " + ", ".join(evicted) + "."
        self._emit(
            "write",
            handle=record.handle,
            object_type=record.object_type,
            explanation=explanation,
            details={
                "mode": record.mode,
                "scope": record.scope,
                "stage": record.stage,
                "produced_by": record.produced_by,
                "derived_from": list(record.derived_from),
                "count": record.count,
                "size_bytes": record.size_bytes,
                "evicted": list(evicted or []),
            },
            duration_ms=duration_ms,
        )

    def rejected(self, record: Recordset, *, reason: str) -> None:
        self._emit(
            "write",
            handle=record.handle,
            object_type=record.object_type,
            explanation=(
                f"Did not store {record.handle} from {record.produced_by}: {reason}. "
                "Nothing downstream will be able to bind to it."
            ),
            details={"admitted": False, "reason": reason, "count": record.count},
        )

    def bound(
        self,
        record: Recordset,
        *,
        tool_name: str,
        explicit: bool,
        candidates: list,
        rule: str,
        duration_ms: int | None = None,
    ) -> None:
        others = [h for h in candidates if h != record.handle]
        if explicit:
            explanation = (
                f"{tool_name} asked for {record.handle} explicitly; bound to it "
                f"({record.count} {record.object_type} from {_origin(record)})."
            )
        else:
            explanation = (
                f"{tool_name} was called without an explicit input. Auto-bound to {record.handle} "
                f"by rule '{rule}' ({record.count} {record.object_type} from {_origin(record)})."
            )
            if others:
                explanation += " Not chosen: " + ", ".join(others) + "."
            else:
                explanation += " It was the only candidate."
        self._emit(
            "bind",
            handle=record.handle,
            object_type=record.object_type,
            explanation=explanation,
            details={
                "tool_name": tool_name,
                "explicit": explicit,
                "rule": rule,
                "candidates": list(candidates),
                "rejected": others,
            },
            duration_ms=duration_ms,
        )

    def bind_failed(
        self, *, object_type: str, tool_name: str, available: list, requested=None
    ) -> None:
        if requested:
            explanation = (
                f"{tool_name} asked for '{requested}', which does not exist in this session."
            )
        else:
            explanation = (
                f"{tool_name} needs a '{object_type}' recordset and none exists in this session."
            )
        explanation += (
            " Available: " + ", ".join(available) + "." if available else " Nothing is stored yet."
        )
        self._emit(
            "bind_failed",
            object_type=object_type,
            explanation=explanation,
            details={"tool_name": tool_name, "requested": requested, "available": list(available)},
        )

    def evicted(
        self, record: Recordset, *, reason: str, tombstoned: bool, dependents: list | None = None
    ) -> None:
        if tombstoned:
            tail = (
                "Tombstoned rather than deleted because {} still records it in derived_from, so "
                "the lineage graph stays intact.".format(", ".join(dependents or []))
            )
        else:
            tail = "Deleted outright; nothing derives from it."
        self._emit(
            "evict",
            handle=record.handle,
            object_type=record.object_type,
            explanation=f"Evicted {record.handle}: {reason}. {tail}",
            details={
                "reason": reason,
                "tombstoned": tombstoned,
                "dependents": list(dependents or []),
                "reclaimed_bytes": record.size_bytes,
            },
        )

    def purged(self, handles: list, *, requested: str, cascaded: list | None = None) -> None:
        explanation = (
            "Governance purge of {requested} removed {n} recordset(s): {handles}. "
            "Payload and metadata both gone from every scope.".format(
                requested=requested, n=len(handles), handles=", ".join(handles) or "none"
            )
        )
        if cascaded:
            explanation += " Cascaded to project-scope copies: " + ", ".join(cascaded) + "."
        self._emit(
            "purge",
            handle=requested,
            explanation=explanation,
            details={"removed": list(handles), "cascaded": list(cascaded or [])},
        )

    def turn_summary(
        self,
        *,
        turn: int,
        created: list,
        read: list,
        evicted: list,
        live_count: int,
        tombstone_count: int,
        total_bytes: int,
    ) -> None:
        """The memory diff — what changed in the store during this turn.

        The single highest-value entry for spotting drift over time, because it
        is comparable across turns without reading anything else.
        """
        explanation = (
            "Turn {turn}: +{created_n} recordset(s){created_list}, {read_n} read, "
            "{evicted_n} evicted. Session now holds {live} live, {tombs} tombstone(s), "
            "{size} of value-mode payload.".format(
                turn=turn,
                created_n=len(created),
                created_list=" (" + ", ".join(created) + ")" if created else "",
                read_n=len(read),
                evicted_n=len(evicted),
                live=live_count,
                tombs=tombstone_count,
                size=human_bytes(total_bytes),
            )
        )
        self._emit(
            "turn_summary",
            explanation=explanation,
            details={
                "turn": turn,
                "created": list(created),
                "read": list(read),
                "evicted": list(evicted),
                "live_count": live_count,
                "tombstone_count": tombstone_count,
                "total_bytes": total_bytes,
            },
        )

    # --- traffic (verbose only) ----------------------------------------

    def read_records(
        self,
        record: Recordset,
        *,
        resolved: int,
        missing: int = 0,
        duration_ms: int | None = None,
    ) -> None:
        if record.is_reference:
            how = "re-fetched from {} index '{}'".format(
                record.ref.get("store", "?"), record.ref.get("index", "?")
            )
        else:
            how = "read from the memory store (value mode, no re-fetch)"
        explanation = f"Resolved {record.handle}: {resolved} {record.object_type} {how}."
        if missing:
            explanation += (
                f" {missing} id(s) were not found — the source documents may have been deleted "
                "since this recordset was written."
            )
        self._emit(
            "read",
            handle=record.handle,
            object_type=record.object_type,
            explanation=explanation,
            details={"resolved": resolved, "missing": missing, "mode": record.mode},
            duration_ms=duration_ms,
        )

    def digested(self, page, *, duration_ms: int | None = None) -> None:
        if page.is_empty:
            explanation = (
                "No recordsets to advertise; the model was told the session has no stored data yet."
            )
        else:
            explanation = (
                "Advertised {shown} of {total} recordset(s) to the model "
                "(page {page}/{pages}): {handles}.".format(
                    shown=page.shown,
                    total=page.total,
                    page=page.page,
                    pages=page.total_pages,
                    handles=", ".join(page.handles),
                )
            )
            if page.total > page.shown:
                explanation += (
                    f" {page.total - page.shown} not shown this page — paginated rather than truncated so the model "
                    "knows more exists."
                )
        self._emit(
            "digest",
            explanation=explanation,
            details={
                "shown": page.shown,
                "total": page.total,
                "page": page.page,
                "total_pages": page.total_pages,
                "handles": list(page.handles),
            },
            duration_ms=duration_ms,
        )


__all__ = [
    "LEVEL_NORMAL",
    "LEVEL_OFF",
    "LEVEL_VERBOSE",
    "ListSink",
    "MemoryOpLog",
    "MemoryOpSink",
    "NullSink",
    "Timer",
]
