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


# --- scripted language model ---------------------------------------------------

_LLM_ENV = (
    "ANATOOLBOX_LLM_BASE_URL",
    "ANATOOLBOX_LLM_API_KEY",
    "ANATOOLBOX_LLM_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


class FakeLLM:
    """An OpenAI-compatible completions endpoint that replays scripted replies.

    Replies are consumed in order. A list reply is streamed item by item; a
    string streams per character when a stream is requested. Every request's
    keyword arguments are kept in ``calls``.
    """

    def __init__(self):
        self.replies = []
        self.calls = []

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if kwargs.get("stream"):
            chunks = reply if isinstance(reply, list) else list(reply)
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=c))])
                for c in chunks
            )
        text = "".join(reply) if isinstance(reply, list) else reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def system_prompt(self, call=0):
        return self.calls[call]["messages"][0]["content"]

    def user_prompt(self, call=0):
        return self.calls[call]["messages"][1]["content"]


@pytest.fixture
def fake_llm(monkeypatch):
    """Route every LLM call to a FakeLLM; roles map to distinct model names."""
    from types import SimpleNamespace

    from anatoolbox import llm_client

    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    llm_client.reset_llm()
    fake = FakeLLM()
    llm_client.set_client_factory(
        lambda **kwargs: SimpleNamespace(kwargs=kwargs, chat=SimpleNamespace(completions=fake))
    )
    llm_client.configure_llm(
        api_key="test",
        model="default-model",
        models={"strong": "strong-model", "fast": "fast-model"},
    )
    yield fake
    llm_client.reset_llm()
