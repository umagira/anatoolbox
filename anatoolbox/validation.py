from __future__ import annotations

import json
from typing import Any

from anatoolbox.base import ToolSchema
from anatoolbox.errors import ToolExecutionError, ToolInputError


def _example_value(prop: dict[str, Any]) -> Any:
    prop_type = prop.get("type", "string")
    if prop_type == "string":
        return "<text>"
    if prop_type == "array":
        item_type = prop.get("items", {}).get("type", "string")
        if item_type == "string":
            return ["<item>"]
        return ["..."]
    if prop_type == "integer":
        return 0
    if prop_type == "boolean":
        return True
    if prop_type == "object":
        return {}
    return "..."


def format_tool_usage(schema: ToolSchema) -> str:
    properties = schema.input_schema.get("properties", {})
    required = schema.input_schema.get("required", [])
    example_args = {name: _example_value(properties.get(name, {})) for name in required}
    return f"/{schema.name} {json.dumps(example_args)}"


def _validate_value(name: str, value: Any, prop: dict[str, Any], *, tool_name: str) -> None:
    expected_type = prop.get("type")
    if expected_type == "string":
        if not isinstance(value, str):
            raise ToolInputError(
                code="invalid_argument_type",
                message=(
                    f"Argument '{name}' must be a string. "
                    f"Example: {format_tool_usage(_schema_with_single_required(tool_name, name, prop))}"
                ),
                tool_name=tool_name,
                details={
                    "argument": name,
                    "expected_type": "string",
                    "received_type": type(value).__name__,
                },
            )
        if not value.strip():
            raise ToolInputError(
                code="invalid_argument_value",
                message=f"Argument '{name}' must be a non-empty string.",
                tool_name=tool_name,
                details={"argument": name},
            )
        return

    if expected_type == "array":
        if not isinstance(value, list):
            raise ToolInputError(
                code="invalid_argument_type",
                message=f"Argument '{name}' must be an array.",
                tool_name=tool_name,
                details={
                    "argument": name,
                    "expected_type": "array",
                    "received_type": type(value).__name__,
                },
            )
        if not value:
            raise ToolInputError(
                code="invalid_argument_value",
                message=f"Argument '{name}' must be a non-empty array.",
                tool_name=tool_name,
                details={"argument": name},
            )
        item_schema = prop.get("items", {})
        item_type = item_schema.get("type")
        if item_type == "string":
            bad = [item for item in value if not isinstance(item, str) or not item.strip()]
            if bad:
                raise ToolInputError(
                    code="invalid_argument_value",
                    message=f"Argument '{name}' must be an array of non-empty strings.",
                    tool_name=tool_name,
                    details={"argument": name},
                )
        return

    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolInputError(
                code="invalid_argument_type",
                message=f"Argument '{name}' must be an integer.",
                tool_name=tool_name,
                details={
                    "argument": name,
                    "expected_type": "integer",
                    "received_type": type(value).__name__,
                },
            )
        return

    if expected_type == "boolean":
        if not isinstance(value, bool):
            raise ToolInputError(
                code="invalid_argument_type",
                message=f"Argument '{name}' must be a boolean.",
                tool_name=tool_name,
                details={
                    "argument": name,
                    "expected_type": "boolean",
                    "received_type": type(value).__name__,
                },
            )


def _schema_with_single_required(tool_name: str, name: str, prop: dict[str, Any]) -> ToolSchema:
    return ToolSchema(
        name=tool_name,
        description="",
        input_schema={"type": "object", "properties": {name: prop}, "required": [name]},
    )


def validate_tool_args(args: Any, schema: ToolSchema) -> None:
    """Validate caller args against a tool's JSON Schema before execute() runs."""
    tool_name = schema.name
    input_schema = schema.input_schema

    if not isinstance(args, dict):
        raise ToolInputError(
            code="invalid_arguments",
            message=f"Tool arguments must be a JSON object. Example: {format_tool_usage(schema)}",
            tool_name=tool_name,
        )

    if input_schema.get("type") not in (None, "object"):
        return

    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    missing = [name for name in required if name not in args]
    if missing:
        missing_label = ", ".join(f"'{name}'" for name in missing)
        text_hint = ""
        if missing == ["query"] and len(required) == 1:
            text_hint = f" Or pass plain text: /{tool_name} your question here"
        raise ToolInputError(
            code="missing_required_arguments",
            message=(
                f"Missing required argument(s): {missing_label}. "
                f"Example: {format_tool_usage(schema)}.{text_hint}"
            ),
            tool_name=tool_name,
            details={"missing": missing, "usage": format_tool_usage(schema)},
        )

    empty_required = [
        name
        for name in required
        if name in args
        and (args[name] is None or _is_empty_value(args[name], properties.get(name, {})))
    ]
    if empty_required:
        empty_label = ", ".join(f"'{name}'" for name in empty_required)
        raise ToolInputError(
            code="invalid_argument_value",
            message=(
                f"Required argument(s) cannot be empty: {empty_label}. "
                f"Example: {format_tool_usage(schema)}"
            ),
            tool_name=tool_name,
            details={"empty": empty_required, "usage": format_tool_usage(schema)},
        )

    for name, value in args.items():
        prop = properties.get(name)
        if prop is None:
            continue
        # Optional JSON-null is omit (retrieve_* query, unused filters). Do not
        # type-error `query: null` as "must be a string".
        if value is None and name not in required:
            continue
        _validate_value(name, value, prop, tool_name=tool_name)


def _is_empty_value(value: Any, prop: dict[str, Any]) -> bool:
    prop_type = prop.get("type", "string")
    if prop_type == "string":
        return not isinstance(value, str) or not value.strip()
    if prop_type == "array":
        return not isinstance(value, list) or not value
    return value is None


def tool_error_result(exc: Exception) -> str:
    if isinstance(exc, ToolExecutionError):
        return exc.to_result()
    return str(exc)
