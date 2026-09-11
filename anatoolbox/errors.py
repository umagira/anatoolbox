from __future__ import annotations

import json
from typing import Any


class ToolExecutionError(Exception):
    """Raised by a Tool.execute() on recoverable failure."""

    def to_result(self) -> str:
        return str(self)


class ToolInputError(ToolExecutionError):
    """Raised before tool execution when caller-supplied arguments are invalid."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        tool_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(message)

    def to_result(self) -> str:
        return json.dumps(
            {
                "error": {
                    "code": self.code,
                    "message": self.message,
                    "tool": self.tool_name,
                    **self.details,
                }
            }
        )

    def __str__(self) -> str:
        return self.to_result()
