import pytest

from anatoolbox import ToolSchema
from anatoolbox.registry import _REGISTRIES


@pytest.fixture(autouse=True)
def clean_registries():
    """Registries are process-global; snapshot and restore around every test.

    The old package leaked module-level state between tests (one suite passed
    alone and failed in-suite). Pin that here rather than rediscover it.
    """
    saved = {name: dict(tools) for name, tools in _REGISTRIES.items()}
    yield
    # Restore *in place*: TOOL_REGISTRY aliases the default dict object, so
    # rebinding it here would silently break that export for later tests.
    for name in list(_REGISTRIES):
        if name not in saved:
            _REGISTRIES[name].clear()
            del _REGISTRIES[name]
    for name, tools in saved.items():
        target = _REGISTRIES.setdefault(name, {})
        target.clear()
        target.update(tools)


def make_tool(name, *, result=None):
    """Minimal object satisfying the dual-use Tool contract."""

    class _Tool:
        schema = ToolSchema(
            name=name,
            description=f"{name} description",
            input_schema={"type": "object", "properties": {}},
        )

        def run(self, args, *, context):
            return dict(result or {"ok": True})

        def execute(self, args, *, context):
            import json

            return json.dumps(self.run(args, context=context))

    return _Tool()


@pytest.fixture
def tool_factory():
    return make_tool
