"""Session-scoped working memory for tools — pointers/handles only, never document bodies.

The agent wires ``Session.working_state`` into ``ToolContext.memory``; notebooks
and pipelines construct ``WorkingMemory()`` (or wrap their own dict) themselves.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any


class WorkingMemory:
    """Small mutable slot store shared across tool calls in one session/run.

    Values must be JSON-friendly dicts (pointers). Large payloads (report
    ``content``, article bodies, etc.) belong in the source system and should
    be re-fetched by id when needed.
    """

    def __init__(self, store: MutableMapping[str, Any] | None = None) -> None:
        self._store: MutableMapping[str, Any] = {} if store is None else store

    @property
    def data(self) -> MutableMapping[str, Any]:
        """Underlying mapping (e.g. ``Session.working_state``)."""
        return self._store

    def get(self, slot: str) -> dict[str, Any] | None:
        value = self._store.get(slot)
        if not isinstance(value, dict):
            return None
        return dict(value)

    def set(self, slot: str, pointer: Mapping[str, Any]) -> None:
        if not isinstance(pointer, Mapping):
            raise TypeError(
                f"WorkingMemory.set({slot!r}) expects a mapping pointer, got {type(pointer)!r}"
            )
        self._store[slot] = dict(pointer)

    def clear(self, slot: str) -> None:
        self._store.pop(slot, None)

    def has(self, slot: str) -> bool:
        return self.get(slot) is not None
