"""The LLM connection: configuration, key resolution, endpoint adaptation, reply parsing."""

from types import SimpleNamespace

import httpx
import openai
import pytest

from anatoolbox import ToolContext
from anatoolbox import llm_client as llm
from anatoolbox.llm_client import ModelSpec, configure_llm, model_for, parse_json_reply
from anatoolbox.llm_tool import LLMTool

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
ENV = (
    "ANATOOLBOX_LLM_BASE_URL",
    "ANATOOLBOX_LLM_API_KEY",
    "ANATOOLBOX_LLM_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _event(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


class FakeCompletions:
    """Replays scripted replies; a list reply streams its items, a str streams per character."""

    def __init__(self):
        self.replies = []
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if kwargs.get("stream"):
            return iter(
                [_event(chunk) for chunk in (reply if isinstance(reply, list) else list(reply))]
            )
        return _response(reply)


def bad_request(message, status=400):
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    body = {"error": {"message": message}}
    response = httpx.Response(status, request=request, json=body)
    cls = openai.BadRequestError if status == 400 else openai.UnprocessableEntityError
    return cls(message, response=response, body=body)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    llm.reset_llm()
    yield
    llm.reset_llm()


@pytest.fixture
def fake():
    completions = FakeCompletions()
    built = []

    def factory(**kwargs):
        client = SimpleNamespace(kwargs=kwargs, chat=SimpleNamespace(completions=completions))
        built.append(client)
        return client

    llm.set_client_factory(factory)
    return SimpleNamespace(completions=completions, built=built)


class TestConfiguration:
    def test_configured_endpoint_key_and_model_reach_the_request(self, fake):
        configure_llm(base_url="http://localhost:11434/v1", api_key="k", model="qwen3:4b")
        fake.completions.replies = ["hi"]
        assert llm.call_llm_text("s", "u") == "hi"
        assert fake.built[0].kwargs["base_url"] == "http://localhost:11434/v1"
        assert fake.built[0].kwargs["api_key"] == "k"
        assert fake.completions.calls[0]["model"] == "qwen3:4b"

    def test_environment_configures_endpoint_and_model(self, fake, monkeypatch):
        monkeypatch.setenv("ANATOOLBOX_LLM_BASE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.setenv("ANATOOLBOX_LLM_MODEL", "local-model")
        fake.completions.replies = ["ok"]
        llm.call_llm_text("s", "u")
        assert fake.completions.calls[0]["model"] == "local-model"
        assert fake.built[0].kwargs["api_key"] == llm.PLACEHOLDER_API_KEY, (
            "local endpoints need no key"
        )

    def test_openai_api_key_is_the_fallback_key(self, fake, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        configure_llm(model="m")
        fake.completions.replies = ["ok"]
        llm.call_llm_text("s", "u")
        assert fake.built[0].kwargs["api_key"] == "sk-test"
        assert "base_url" not in fake.built[0].kwargs

    def test_openai_without_a_key_is_explained(self, fake):
        configure_llm(model="m")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm.call_llm_text("s", "u")

    def test_a_missing_model_is_explained(self, fake):
        configure_llm(api_key="k")
        with pytest.raises(RuntimeError, match="No language model configured"):
            llm.call_llm_text("s", "u")

    def test_roles_fall_back_to_the_default_model(self):
        configure_llm(model="base", models={"strong": "big"})
        assert model_for("strong") == "big"
        assert model_for("fast") == "base"

    def test_an_explicit_model_wins(self, fake):
        configure_llm(api_key="k", model="base")
        fake.completions.replies = ["ok"]
        llm.call_llm_text("s", "u", model="other")
        assert fake.completions.calls[0]["model"] == "other"

    def test_configure_keeps_unspecified_values(self):
        configure_llm(base_url="http://x/v1", model="m")
        configure_llm(api_key="k")
        settings = llm.llm_settings()
        assert (settings.base_url, settings.api_key, settings.models["default"]) == (
            "http://x/v1",
            "k",
            "m",
        )

    def test_reset_forgets_configuration(self):
        configure_llm(model="m")
        llm.reset_llm()
        with pytest.raises(RuntimeError):
            model_for()

    def test_invalid_json_mode_is_rejected(self):
        with pytest.raises(ValueError, match="json_mode"):
            ModelSpec(json_mode="yaml")


class TestKeys:
    def test_project_resolver_and_explicit_key(self, fake):
        configure_llm(model="m")
        llm.configure_openai_key_resolver(lambda project: f"key-{project}")
        fake.completions.replies = ["a", "b"]
        llm.call_llm_text("s", "u", project="alpha")
        llm.call_llm_text("s", "u", project="alpha", api_key="direct")
        assert [c.kwargs["api_key"] for c in fake.built] == ["key-alpha", "direct"]

    def test_a_resolver_returning_nothing_falls_through(self, fake):
        configure_llm(model="m", api_key="configured")
        llm.configure_openai_key_resolver(lambda project: "")
        fake.completions.replies = ["a"]
        llm.call_llm_text("s", "u", project="alpha")
        assert fake.built[0].kwargs["api_key"] == "configured"

    def test_clients_are_cached_per_endpoint_and_key(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = ["a", "b"]
        llm.call_llm_text("s", "u")
        llm.call_llm_text("s", "u")
        assert len(fake.built) == 1


class TestEndpointAdaptation:
    def test_temperature_is_omitted_when_the_spec_says_so(self, fake):
        configure_llm(api_key="k", model="m", specs={"m": ModelSpec(temperature=False)})
        fake.completions.replies = ["ok"]
        llm.call_llm_text("s", "u")
        assert "temperature" not in fake.completions.calls[0]

    def test_a_rejected_temperature_is_dropped_and_remembered(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [
            bad_request("Unsupported value: 'temperature' does not support 0.0"),
            "ok",
            "again",
        ]
        assert llm.call_llm_text("s", "u") == "ok"
        assert llm.call_llm_text("s", "u") == "again"
        assert ["temperature" in c for c in fake.completions.calls] == [True, False, False]

    def test_a_rejected_reasoning_effort_is_dropped(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [
            bad_request("Unrecognized request argument: reasoning_effort", 422),
            "ok",
        ]
        assert llm.call_llm_text("s", "u", reasoning_effort="low") == "ok"
        assert "reasoning_effort" not in fake.completions.calls[1]

    def test_spec_extra_body_and_reasoning_effort_are_sent(self, fake):
        spec = ModelSpec(
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}, reasoning_effort="low"
        )
        configure_llm(api_key="k", model="m", specs={"m": spec})
        fake.completions.replies = ["ok"]
        llm.call_llm_text("s", "u")
        call = fake.completions.calls[0]
        assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        assert call["reasoning_effort"] == "low"

    def test_unrelated_errors_are_not_swallowed(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [
            bad_request("This model's maximum context length is 8192 tokens")
        ]
        with pytest.raises(openai.BadRequestError):
            llm.call_llm_text("s", "u")


class TestJson:
    def test_a_schema_asks_for_strict_structured_output_first(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = ['{"a": 1}']
        assert llm.call_llm_json("Return it.", "u", response_schema=SCHEMA) == {"a": 1}
        assert fake.completions.calls[0]["response_format"]["type"] == "json_schema"

    def test_json_formats_degrade_until_the_endpoint_accepts(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [
            bad_request("response_format type json_schema is not supported"),
            bad_request("response_format type json_object is not supported"),
            'Sure! ```json\n{"a": 2}\n```',
            '{"a": 3}',
        ]
        assert llm.call_llm_json("Return it.", "u", response_schema=SCHEMA) == {"a": 2}
        calls = fake.completions.calls
        assert calls[1]["response_format"] == {"type": "json_object"}
        assert "response_format" not in calls[2]
        assert '"required": ["a"]' in calls[2]["messages"][0]["content"], (
            "schema moves into the prompt"
        )
        assert llm.call_llm_json("Return it.", "u", response_schema=SCHEMA) == {"a": 3}
        assert "response_format" not in calls[3], "the downgrade is remembered"

    def test_without_a_schema_json_mode_is_used(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = ['{"a": 1}']
        llm.call_llm_json("Return it.", "u")
        assert fake.completions.calls[0]["response_format"] == {"type": "json_object"}

    def test_an_empty_json_reply_raises(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [""]
        with pytest.raises(ValueError, match="empty"):
            llm.call_llm_json("Return it.", "u")

    @pytest.mark.parametrize(
        "reply, expected",
        [
            ('{"a": 1}', {"a": 1}),
            ('```json\n{"a": 1}\n```', {"a": 1}),
            ('<think>maybe {"b": 0}</think>\n{"a": 1}', {"a": 1}),
            ('Here you go: {"a": 1} — hope that helps', {"a": 1}),
            ('[{"a": 1}]', [{"a": 1}]),
        ],
    )
    def test_parse_json_reply(self, reply, expected):
        assert parse_json_reply(reply) == expected

    def test_a_reply_without_json_is_explained(self):
        with pytest.raises(ValueError, match="Could not find JSON"):
            parse_json_reply("no structured data here")


class TestText:
    def test_a_leading_reasoning_block_is_removed(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = ["<think>plan the answer</think>\n\nAnswer."]
        assert llm.call_llm_text("s", "u") == "Answer."

    def test_reasoning_tags_split_across_chunks_are_removed(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [["<th", "ink>a", "b</thi", "nk>", " Hel", "lo"]]
        assert "".join(llm.call_llm_text_stream("s", "u")) == "Hello"

    def test_whitespace_only_chunks_after_reasoning_are_dropped(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [["<think>x</think>", "\n", "  ", "\nAnswer", " ends."]]
        assert llm.call_llm_text("s", "u") == "Answer ends."

    def test_text_that_merely_starts_with_a_tag_is_kept(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [["<b>bold</b>", " text"]]
        assert llm.call_llm_text("s", "u") == "<b>bold</b> text"

    def test_an_empty_text_reply_raises(self, fake):
        configure_llm(api_key="k", model="m")
        fake.completions.replies = [["<think>only thinking</think>"]]
        with pytest.raises(ValueError, match="empty"):
            llm.call_llm_text("s", "u")


def test_llm_tool_uses_its_configured_role(fake):
    class StrongTool(LLMTool):
        role = "strong"
        system_prompt = "Return JSON."

        def _build_user_prompt(self, args):
            return "x"

    configure_llm(api_key="k", model="base", models={"strong": "big"})
    fake.completions.replies = ['{"ok": true}']
    assert StrongTool().run({}, context=ToolContext(project="p", session_id="s")) == {"ok": True}
    assert fake.completions.calls[0]["model"] == "big"
