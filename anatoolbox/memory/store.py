"""MemoryStore — where recordsets live. Mechanism only, no policy.

The seam between this module and ``policy.py`` is deliberate: a store knows
how to put bytes somewhere and get them back, and nothing else. It never
decides what is worth keeping, what to show the model, or when to compact.
That keeps the two axes independently swappable — any store against any
policy — which is the point of ``AgentConfig.memory_store`` and
``AgentConfig.memory_policy`` being separate fields.

Three implementations ship here:

``JsonFileRecordsetStore``
    Default. One JSON file per recordset, mirroring ``JsonFileArtifactStore``
    in the agent package. Survives restart, needs no infrastructure, and the
    filesystem is the only shared state so two instances pointed at the same
    directory stay consistent for free.

``InMemoryRecordsetStore``
    Tests and notebooks. Same semantics, no disk.

``NullRecordsetStore``
    Accepts writes and forgets them. This is the memory ablation: set
    ``memory_store = "none"`` on a project to measure what memory is actually
    buying you, without touching any other code.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Protocol

from anatoolbox.memory.recordset import PROJECT_SCOPE, SESSION_SCOPE, Recordset


class MemoryStore(Protocol):
    """Storage mechanism for recordsets. Narrow on purpose."""

    def allocate_handle(self, *, owner: str, object_type: str) -> str:
        """Reserve the next handle for an object_type, e.g. ``articles_3``.

        Ordinals never repeat within an owner, even after deletion, so a
        handle mentioned in an old conversation turn or log line can never
        silently point at different data later.
        """
        ...

    def write(self, record: Recordset) -> str: ...

    def read(self, handle: str, *, owner: str) -> Recordset | None: ...

    def query(
        self,
        *,
        owner: str,
        object_type: str | None = None,
        limit: int | None = None,
        include_tombstoned: bool = False,
    ) -> list:
        """Newest-first recordsets for an owner, optionally one object_type."""
        ...

    def delete(self, handle: str, *, owner: str, tombstone: bool = False) -> bool:
        """Remove a recordset, or replace it with a metadata-only tombstone.

        Returns True when something was actually removed or tombstoned.
        """
        ...


def owner_key(*, scope: str, session_id: str, project: str) -> str:
    """Ownership key for a scope: sessions are private, projects are shared."""
    if scope == PROJECT_SCOPE:
        return f"project:{project}"
    return f"session:{session_id}"


_HANDLE_ORDINAL = re.compile(r"^(?P<base>.+)_(?P<ordinal>\d+)$")


def _next_ordinal(existing_handles, object_type: str) -> int:
    """Highest used ordinal for object_type, plus one."""
    highest = 0
    for handle in existing_handles:
        match = _HANDLE_ORDINAL.match(handle)
        if match is None or match.group("base") != object_type:
            continue
        highest = max(highest, int(match.group("ordinal")))
    return highest + 1


def _sort_newest_first(records: list) -> list:
    """Newest first, tie-broken by handle ordinal rather than lexically.

    Timestamps collide inside a single turn, and a string fallback would rank
    articles_9 above articles_10.
    """
    return sorted(records, key=lambda r: (r.created_at, r.ordinal), reverse=True)


def _apply_query(
    records: list,
    *,
    object_type: str | None,
    limit: int | None,
    include_tombstoned: bool,
) -> list:
    out = [
        r
        for r in records
        if (object_type is None or r.object_type == object_type)
        and (include_tombstoned or not r.tombstoned)
    ]
    out = _sort_newest_first(out)
    if limit is not None:
        out = out[:limit]
    return out


class InMemoryRecordsetStore:
    """Process-local MemoryStore. Same semantics as the file store, no disk."""

    def __init__(self) -> None:
        # owner -> handle -> Recordset
        self._records: dict = {}
        # owner -> every handle ever allocated, so ordinals never repeat
        self._allocated: dict = {}
        self._lock = threading.Lock()

    def allocate_handle(self, *, owner: str, object_type: str) -> str:
        with self._lock:
            seen = self._allocated.setdefault(owner, set())
            handle = f"{object_type}_{_next_ordinal(seen, object_type)}"
            seen.add(handle)
            return handle

    def write(self, record: Recordset) -> str:
        owner = owner_key(scope=record.scope, session_id=record.session_id, project=record.project)
        with self._lock:
            self._records.setdefault(owner, {})[record.handle] = record
            self._allocated.setdefault(owner, set()).add(record.handle)
        return record.handle

    def read(self, handle: str, *, owner: str) -> Recordset | None:
        with self._lock:
            return self._records.get(owner, {}).get(handle)

    def query(
        self,
        *,
        owner: str,
        object_type: str | None = None,
        limit: int | None = None,
        include_tombstoned: bool = False,
    ) -> list:
        with self._lock:
            records = list(self._records.get(owner, {}).values())
        return _apply_query(
            records, object_type=object_type, limit=limit, include_tombstoned=include_tombstoned
        )

    def delete(self, handle: str, *, owner: str, tombstone: bool = False) -> bool:
        with self._lock:
            bucket = self._records.get(owner, {})
            record = bucket.get(handle)
            if record is None:
                return False
            if tombstone:
                bucket[handle] = record.tombstone()
            else:
                del bucket[handle]
            return True


class JsonFileRecordsetStore:
    """One JSON file per recordset at ``store_dir/<owner>__<handle>.json``.

    Mirrors ``JsonFileArtifactStore``: no in-process cache, so every call
    re-reads the file and multiple instances pointed at the same directory
    stay consistent. The owner prefix is part of the filename because handles
    are only unique within an owner — ``articles_1`` exists in every session.

    Handle allocation reads the ``.allocated`` sidecar rather than the live
    files, so a deleted ``articles_2`` does not free the name for reuse.
    """

    def __init__(self, store_dir: Path) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _safe(token: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", token)

    def _path(self, owner: str, handle: str) -> Path:
        return self._dir / f"{self._safe(owner)}__{self._safe(handle)}.json"

    def _allocated_path(self, owner: str) -> Path:
        return self._dir / f"{self._safe(owner)}.allocated.json"

    def _read_allocated(self, owner: str) -> list:
        path = self._allocated_path(owner)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        return [str(h) for h in data] if isinstance(data, list) else []

    def allocate_handle(self, *, owner: str, object_type: str) -> str:
        with self._lock:
            seen = self._read_allocated(owner)
            handle = f"{object_type}_{_next_ordinal(seen, object_type)}"
            seen.append(handle)
            self._allocated_path(owner).write_text(json.dumps(seen))
            return handle

    def write(self, record: Recordset) -> str:
        owner = owner_key(scope=record.scope, session_id=record.session_id, project=record.project)
        with self._lock:
            self._path(owner, record.handle).write_text(
                json.dumps(record.to_dict(), indent=2, default=str)
            )
            seen = self._read_allocated(owner)
            if record.handle not in seen:
                seen.append(record.handle)
                self._allocated_path(owner).write_text(json.dumps(seen))
        return record.handle

    def read(self, handle: str, *, owner: str) -> Recordset | None:
        path = self._path(owner, handle)
        if not path.exists():
            return None
        try:
            return Recordset.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            return None

    def query(
        self,
        *,
        owner: str,
        object_type: str | None = None,
        limit: int | None = None,
        include_tombstoned: bool = False,
    ) -> list:
        prefix = f"{self._safe(owner)}__"
        records = []
        for path in self._dir.glob(f"{prefix}*.json"):
            try:
                records.append(Recordset.from_dict(json.loads(path.read_text())))
            except (OSError, ValueError, TypeError):
                continue
        return _apply_query(
            records, object_type=object_type, limit=limit, include_tombstoned=include_tombstoned
        )

    def delete(self, handle: str, *, owner: str, tombstone: bool = False) -> bool:
        record = self.read(handle, owner=owner)
        if record is None:
            return False
        with self._lock:
            if tombstone:
                self._path(owner, handle).write_text(
                    json.dumps(record.tombstone().to_dict(), indent=2, default=str)
                )
            else:
                self._path(owner, handle).unlink(missing_ok=True)
            return True


class NullRecordsetStore:
    """Accepts writes and forgets them — the memory ablation backend.

    Handles are still allocated so producing tools return something coherent,
    but nothing is retrievable, so every downstream bind fails the same way it
    would if the producing tool had never run. That is the honest ablation:
    memory absent, not memory silently degraded.
    """

    def __init__(self) -> None:
        self._counters: dict = {}
        self._lock = threading.Lock()

    def allocate_handle(self, *, owner: str, object_type: str) -> str:
        with self._lock:
            key = (owner, object_type)
            self._counters[key] = self._counters.get(key, 0) + 1
            return f"{object_type}_{self._counters[key]}"

    def write(self, record: Recordset) -> str:
        return record.handle

    def read(self, handle: str, *, owner: str) -> Recordset | None:
        return None

    def query(
        self,
        *,
        owner: str,
        object_type: str | None = None,
        limit: int | None = None,
        include_tombstoned: bool = False,
    ) -> list:
        return []

    def delete(self, handle: str, *, owner: str, tombstone: bool = False) -> bool:
        return False


__all__ = [
    "PROJECT_SCOPE",
    "SESSION_SCOPE",
    "InMemoryRecordsetStore",
    "JsonFileRecordsetStore",
    "MemoryStore",
    "NullRecordsetStore",
    "owner_key",
]
