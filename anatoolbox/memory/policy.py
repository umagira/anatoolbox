"""MemoryPolicy — the decisions a store must not make.

Storage verbs (write/read/query/delete) are mechanism. What is worth keeping,
what the model gets to see, and when to compact are judgment calls, and they
belong on their own axis so a policy can be swapped or A/B tested without
rewriting a backend.

The digest is the load-bearing piece. A recordset the model cannot see is a
recordset the model will never bind to, so the digest is what turns storage
into usable memory. It **paginates rather than truncates**: a truncated list
silently hides data, whereas a ``Showing 3 of 47`` header tells the model more
exists and roughly how much. It carries handles, volumes and provenance —
never records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from anatoolbox.memory.oplog import human_bytes
from anatoolbox.memory.recordset import Recordset

DEFAULT_PAGE_SIZE = 12


@dataclass
class DigestPage:
    """One page of the memory digest, ready to drop into a system prompt."""

    text: str
    page: int
    total_pages: int
    shown: int
    total: int
    handles: list = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return self.total == 0


class MemoryPolicy(Protocol):
    """Decisions over a MemoryStore. Swappable independently of storage."""

    page_size: int

    def admit(self, candidate: Recordset) -> bool:
        """Write-path filter: is this worth storing at all?"""
        ...

    def digest(self, available: list, *, page: int = 1) -> DigestPage:
        """Render one page of what the model can bind to."""
        ...

    def evictable(self, available: list, *, now: datetime | None = None) -> list:
        """Recordsets this policy considers stale, with the reason for each.

        Returns ``(recordset, reason)`` pairs; the reason is logged verbatim so
        an eviction is never unexplained in the audit trail.
        """
        ...


def _age_phrase(created_at: str, *, now: datetime | None = None) -> str:
    seconds = _age_seconds(created_at, now=now)
    if seconds is None:
        return "age unknown"
    if seconds < 45:
        return "just now"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{max(minutes, 1)} min ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours} h ago"
    return f"{int(hours // 24)} d ago"


def _age_seconds(created_at: str, *, now: datetime | None = None) -> float | None:
    try:
        created = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return (reference - created).total_seconds()


def _size_phrase(record: Recordset) -> str:
    """Size awareness in the digest: the model should know what it is asking for."""
    if record.is_reference:
        return "by ref"
    return human_bytes(record.size_bytes)


class DefaultMemoryPolicy:
    """Keep everything a tool produced; page the digest; evict on age and count.

    Deliberately conservative on the write path. In a staged research pipeline
    every tool output is a deliberate act by the model, so there is no noise to
    filter the way a chat transcript has — the survey's write-path filtering
    argument applies to conversational memory, not to this. Empty recordsets
    are the one exception: they carry no records and would only crowd the
    digest.
    """

    def __init__(
        self,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_age_seconds: float | None = None,
        max_per_object_type: int | None = None,
    ) -> None:
        self.page_size = page_size
        self.max_age_seconds = max_age_seconds
        self.max_per_object_type = max_per_object_type

    def admit(self, candidate: Recordset) -> bool:
        return candidate.count > 0

    def digest(self, available: list, *, page: int = 1) -> DigestPage:
        live = [r for r in available if not r.tombstoned]
        total = len(live)
        if total == 0:
            return DigestPage(
                text="No stored data in this session yet.",
                page=1,
                total_pages=1,
                shown=0,
                total=0,
                handles=[],
            )

        total_pages = max(1, -(-total // self.page_size))
        page = min(max(page, 1), total_pages)
        start = (page - 1) * self.page_size
        window = live[start : start + self.page_size]

        header = f"Stored data you can pass as input= (showing {len(window)} of {total}"
        header += f", page {page}/{total_pages}):" if total_pages > 1 else "):"

        lines = [header]
        for record in window:
            lines.append(
                f"  {record.handle} | {record.count} {record.object_type} | {_size_phrase(record)} | {record.origin()} | {_age_phrase(record.created_at)}"
            )
        if total_pages > 1:
            lines.append(
                f"  ({total - len(window)} more not shown — ask for the next page if needed.)"
            )
        return DigestPage(
            text="\n".join(lines),
            page=page,
            total_pages=total_pages,
            shown=len(window),
            total=total,
            handles=[r.handle for r in window],
        )

    def evictable(self, available: list, *, now: datetime | None = None) -> list:
        out = []
        if self.max_age_seconds is not None:
            for record in available:
                if record.tombstoned:
                    continue
                age = _age_seconds(record.created_at, now=now)
                if age is not None and age > self.max_age_seconds:
                    out.append(
                        (
                            record,
                            f"age {age:.0f}s exceeds max_age_seconds={self.max_age_seconds:.0f}",
                        )
                    )
        if self.max_per_object_type is not None:
            by_type: dict = {}
            for record in available:
                if record.tombstoned:
                    continue
                by_type.setdefault(record.object_type, []).append(record)
            for object_type, records in by_type.items():
                ordered = sorted(records, key=lambda r: r.created_at, reverse=True)
                for record in ordered[self.max_per_object_type :]:
                    out.append(
                        (
                            record,
                            f"more than max_per_object_type={self.max_per_object_type} recordsets of '{object_type}'",
                        )
                    )
        # De-duplicate while keeping the first reason recorded for each handle.
        seen = set()
        deduped = []
        for record, reason in out:
            if record.handle in seen:
                continue
            seen.add(record.handle)
            deduped.append((record, reason))
        return deduped


class RetainAllMemoryPolicy(DefaultMemoryPolicy):
    """Never evict. Useful as an A/B baseline against an eviction policy."""

    def __init__(self, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        super().__init__(page_size=page_size, max_age_seconds=None, max_per_object_type=None)

    def admit(self, candidate: Recordset) -> bool:
        return True

    def evictable(self, available: list, *, now: datetime | None = None) -> list:
        return []


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "DefaultMemoryPolicy",
    "DigestPage",
    "MemoryPolicy",
    "RetainAllMemoryPolicy",
]
