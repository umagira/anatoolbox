"""Language model access for tools — any OpenAI-compatible endpoint.

Tools that need a language model call this module, never a vendor SDK
directly. It speaks the OpenAI chat-completions protocol, which OpenAI and
most other providers and local servers implement — Ollama, vLLM, LM Studio,
llama.cpp's server, OpenRouter, Together, Groq, Gemini's OpenAI-compatible
endpoint — so choosing a model is configuration, not code::

    from anatoolbox.llm_client import configure_llm

    configure_llm(model="gpt-4o-mini")                                   # OpenAI; key from OPENAI_API_KEY
    configure_llm(base_url="http://localhost:11434/v1", model="qwen3:4b")  # Ollama on your machine
    configure_llm(base_url="https://openrouter.ai/api/v1", api_key="...", model="qwen/qwen3-8b")

The same can come from the environment: ``ANATOOLBOX_LLM_BASE_URL``,
``ANATOOLBOX_LLM_API_KEY`` and ``ANATOOLBOX_LLM_MODEL`` (the OpenAI SDK's
``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` are honoured as fallbacks).

There is no default model on purpose: which model produced a result is part
of the result.

**Roles.** Tools ask for a role (``"default"``, ``"fast"``, ``"strong"``)
rather than a model name; ``configure_llm(models={...})`` maps roles to
models, and an unmapped role uses the default model.

**Endpoints differ.** Rather than guessing what a model supports from its
name, this module adapts to what the endpoint actually accepts. A parameter
the endpoint rejects (``temperature``, ``reasoning_effort``, a strict JSON
schema) is dropped, the request retried, and the lesson remembered for that
model. JSON requests degrade from a strict schema, to JSON mode, to a
prompt-only request whose JSON is extracted from the reply — which also
copes with reasoning models that think aloud in ``<think>`` blocks. Declare
known quirks up front with ``ModelSpec`` to skip the failed first attempt.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

ENV_BASE_URL = "ANATOOLBOX_LLM_BASE_URL"
ENV_API_KEY = "ANATOOLBOX_LLM_API_KEY"
ENV_MODEL = "ANATOOLBOX_LLM_MODEL"

#: Sent to a custom endpoint that needs no key; the OpenAI SDK refuses an empty one.
PLACEHOLDER_API_KEY = "not-needed"

JSON_SCHEMA = "json_schema"
JSON_OBJECT = "json_object"
JSON_NONE = "none"
_JSON_MODES = (JSON_SCHEMA, JSON_OBJECT, JSON_NONE)


@dataclass(frozen=True)
class ModelSpec:
    """What a model on your endpoint accepts. Declare it to skip discovery.

    ``json_mode`` is the strongest JSON enforcement the endpoint supports:
    ``"json_schema"`` (strict structured output), ``"json_object"`` (JSON
    mode) or ``"none"`` (instructions in the prompt only). ``extra_body`` is
    passed through verbatim for provider-specific options, for example
    ``{"chat_template_kwargs": {"enable_thinking": False}}`` on vLLM.
    """

    temperature: bool = True
    json_mode: str = JSON_SCHEMA
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.json_mode not in _JSON_MODES:
            raise ValueError(f"json_mode must be one of {_JSON_MODES}, got {self.json_mode!r}.")


@dataclass
class LLMSettings:
    base_url: str | None = None
    api_key: str | None = None
    models: dict[str, str] = field(default_factory=dict)
    specs: dict[str, ModelSpec] = field(default_factory=dict)
    timeout: float = 120.0
    max_retries: int = 2


_SETTINGS = LLMSettings()
_CLIENTS: dict[tuple, Any] = {}
_LEARNED: dict[str, ModelSpec] = {}
_KEY_RESOLVER: Callable[[str], str] | None = None
_CLIENT_FACTORY: Callable[..., Any] | None = None


# --- configuration ---------------------------------------------------------


def configure_llm(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    models: dict[str, str] | None = None,
    specs: dict[str, ModelSpec] | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> LLMSettings:
    """Point tools at an endpoint and model. Arguments left as None keep their value.

    ``model`` sets the default role; ``models`` maps further roles, e.g.
    ``{"fast": "qwen3:1.7b", "strong": "qwen3:14b"}``. Returns the effective
    settings, environment included.
    """
    global _SETTINGS
    updated = replace(_SETTINGS, models=dict(_SETTINGS.models), specs=dict(_SETTINGS.specs))
    if base_url is not None:
        updated.base_url = base_url.strip() or None
    if api_key is not None:
        updated.api_key = api_key.strip() or None
    if models:
        updated.models.update(models)
    if model is not None:
        updated.models["default"] = model
    if specs:
        updated.specs.update(specs)
    if timeout is not None:
        updated.timeout = float(timeout)
    if max_retries is not None:
        updated.max_retries = int(max_retries)
    _SETTINGS = updated
    _CLIENTS.clear()
    _LEARNED.clear()
    return llm_settings()


def reset_llm() -> None:
    """Forget all configuration, cached clients, and learned model quirks."""
    global _SETTINGS, _KEY_RESOLVER, _CLIENT_FACTORY
    _SETTINGS = LLMSettings()
    _KEY_RESOLVER = None
    _CLIENT_FACTORY = None
    _CLIENTS.clear()
    _LEARNED.clear()


def llm_settings() -> LLMSettings:
    """The effective settings: configured values first, then the environment."""
    models = dict(_SETTINGS.models)
    if "default" not in models and _env(ENV_MODEL):
        models["default"] = _env(ENV_MODEL)
    return LLMSettings(
        base_url=_SETTINGS.base_url or _env(ENV_BASE_URL) or _env("OPENAI_BASE_URL"),
        api_key=_SETTINGS.api_key or _env(ENV_API_KEY),
        models=models,
        specs=dict(_SETTINGS.specs),
        timeout=_SETTINGS.timeout,
        max_retries=_SETTINGS.max_retries,
    )


def model_for(role: str = "default") -> str:
    """The model name a role maps to; unmapped roles use the default model."""
    models = llm_settings().models
    name = models.get(role) or models.get("default")
    if not name:
        raise RuntimeError(
            "No language model configured. Call "
            "anatoolbox.llm_client.configure_llm(model='...') or set ANATOOLBOX_LLM_MODEL. "
            "There is no default model on purpose: which model produced a result is part "
            "of the result."
        )
    return name


def spec_for(model: str) -> ModelSpec:
    """What is known about a model: learned quirks, else the declared spec."""
    return _LEARNED.get(model) or llm_settings().specs.get(model) or ModelSpec()


def configure_openai_key_resolver(resolver: Callable[[str], str] | None) -> None:
    """Resolve the API key per project (the ``project=`` of each call).

    Optional: a host serving several projects with separate keys installs
    one. A resolver returning an empty value falls through to the configured
    or environment key.
    """
    global _KEY_RESOLVER
    _KEY_RESOLVER = resolver
    _CLIENTS.clear()


def set_client_factory(factory: Callable[..., Any] | None) -> None:
    """Replace how clients are built — for tests, tracing, or an SDK with the same interface.

    The factory receives the OpenAI client's keyword arguments (``api_key``,
    ``timeout``, ``max_retries`` and, for custom endpoints, ``base_url``).
    """
    global _CLIENT_FACTORY
    _CLIENT_FACTORY = factory
    _CLIENTS.clear()


def clear_openai_clients() -> None:
    """Drop cached clients (they are rebuilt on the next call)."""
    _CLIENTS.clear()


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _resolve_api_key(*, project: str | None, api_key: str | None, base_url: str | None) -> str:
    if api_key and str(api_key).strip():
        return str(api_key).strip()
    if _KEY_RESOLVER is not None and project and str(project).strip():
        resolved = _KEY_RESOLVER(str(project).strip())
        if resolved and str(resolved).strip():
            return str(resolved).strip()
    configured = llm_settings().api_key or _env("OPENAI_API_KEY")
    if configured:
        return configured
    if base_url:
        return PLACEHOLDER_API_KEY
    raise RuntimeError(
        "No API key for the OpenAI API. Set OPENAI_API_KEY, call "
        "configure_llm(api_key=...), or point configure_llm(base_url=...) at an "
        "endpoint that needs no key, such as a local Ollama or vLLM server."
    )


def _client(*, project: str | None, api_key: str | None) -> Any:
    settings = llm_settings()
    key = _resolve_api_key(project=project, api_key=api_key, base_url=settings.base_url)
    cache_key = (settings.base_url, key)
    client = _CLIENTS.get(cache_key)
    if client is None:
        kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": settings.timeout,
            "max_retries": settings.max_retries,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        if _CLIENT_FACTORY is not None:
            client = _CLIENT_FACTORY(**kwargs)
        else:
            from openai import OpenAI

            client = OpenAI(**kwargs)
        _CLIENTS[cache_key] = client
    return client


# --- requests --------------------------------------------------------------


class _FormatRejected(Exception):
    """The endpoint refused the requested response_format."""


def _rejected_parameter(exc: Exception) -> str | None:
    """Name the request parameter an endpoint rejected, when its error says so."""
    if getattr(exc, "status_code", None) not in (400, 422):
        return None
    message = str(getattr(exc, "message", "") or exc).lower()
    body = getattr(exc, "body", None)
    if body:
        message += " " + json.dumps(body, default=str).lower()
    for parameter in ("temperature", "reasoning_effort", "response_format"):
        if parameter in message:
            return parameter
    if JSON_SCHEMA in message or JSON_OBJECT in message:
        return "response_format"
    return None


def _create(client: Any, model: str, kwargs: dict[str, Any]) -> Any:
    """chat.completions.create, dropping parameters the endpoint rejects (and remembering)."""
    while True:
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            parameter = _rejected_parameter(exc)
            if parameter in ("temperature", "reasoning_effort") and parameter in kwargs:
                learned = (
                    {"temperature": False}
                    if parameter == "temperature"
                    else {"reasoning_effort": None}
                )
                _LEARNED[model] = replace(spec_for(model), **learned)
                kwargs = {k: v for k, v in kwargs.items() if k != parameter}
                continue
            if parameter == "response_format" and "response_format" in kwargs:
                raise _FormatRejected(kwargs["response_format"].get("type")) from exc
            raise


def _base_kwargs(
    model: str,
    spec: ModelSpec,
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    reasoning_effort: str | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None and spec.temperature:
        kwargs["temperature"] = temperature
    effort = reasoning_effort if reasoning_effort is not None else spec.reasoning_effort
    if effort:
        kwargs["reasoning_effort"] = effort
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if spec.extra_body:
        kwargs["extra_body"] = dict(spec.extra_body)
    return kwargs


def _messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    role: str = "default",
    temperature: float | None = 0.0,
    response_schema: dict[str, Any] | None = None,
    project: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> Any:
    """One structured-JSON completion. Tool-internal: the orchestrating model never sees it.

    With ``response_schema``, asks for strict structured output first and
    falls back as the endpoint requires (see the module docstring). Without
    it, uses JSON mode. In the weaker modes the schema is written into the
    system prompt so the model still knows the shape.
    """
    name = model or model_for(role)
    client = _client(project=project, api_key=api_key)
    while True:
        spec = spec_for(name)
        mode = spec.json_mode
        if response_schema is None and mode == JSON_SCHEMA:
            mode = JSON_OBJECT
        system = system_prompt
        if mode != JSON_SCHEMA:
            system += "\n\nRespond with a single JSON object and nothing else."
            if response_schema is not None:
                system += " It must match this JSON Schema:\n" + json.dumps(response_schema)
        kwargs = _base_kwargs(
            name,
            spec,
            _messages(system, user_prompt),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        if mode == JSON_SCHEMA:
            kwargs["response_format"] = {
                "type": JSON_SCHEMA,
                "json_schema": {"name": "response", "strict": True, "schema": response_schema},
            }
        elif mode == JSON_OBJECT:
            kwargs["response_format"] = {"type": JSON_OBJECT}
        try:
            response = _create(client, name, kwargs)
        except _FormatRejected as rejected:
            downgraded = JSON_OBJECT if str(rejected) == JSON_SCHEMA else JSON_NONE
            _LEARNED[name] = replace(spec_for(name), json_mode=downgraded)
            continue
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("LLM returned an empty response")
        return parse_json_reply(content)


def call_llm_text_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    role: str = "default",
    temperature: float | None = 0.0,
    project: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    """Stream a plain-text completion, without any leading ``<think>`` block."""
    name = model or model_for(role)
    kwargs = _base_kwargs(
        name,
        spec_for(name),
        _messages(system_prompt, user_prompt),
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
    )
    kwargs["stream"] = True
    stream = _create(_client(project=project, api_key=api_key), name, kwargs)
    yield from _without_reasoning(_deltas(stream))


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    role: str = "default",
    temperature: float | None = 0.0,
    project: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """A plain-text completion, collected from ``call_llm_text_stream``."""
    content = "".join(
        call_llm_text_stream(
            system_prompt,
            user_prompt,
            model=model,
            role=role,
            temperature=temperature,
            project=project,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
    )
    if not content.strip():
        raise ValueError("LLM returned an empty response")
    return content


# --- reply handling ----------------------------------------------------------

_OPEN_THINK, _CLOSE_THINK = "<think>", "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """Remove ``<think>…</think>`` blocks that reasoning models emit before answering."""
    return _THINK_BLOCK.sub("", text or "").strip()


def parse_json_reply(text: str) -> Any:
    """The JSON value in a model reply: bare, fenced, or surrounded by prose."""
    cleaned = strip_reasoning(text)
    decoder = json.JSONDecoder()
    for candidate in [m.group(1) for m in _FENCE.finditer(cleaned)] + [cleaned]:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except ValueError:
            pass
        for position, char in enumerate(candidate):
            if char in "{[":
                try:
                    return decoder.raw_decode(candidate, position)[0]
                except ValueError:
                    continue
    raise ValueError(f"Could not find JSON in the model's reply: {cleaned[:200]!r}")


def _deltas(stream: Iterable[Any]) -> Iterator[str]:
    for event in stream:
        choices = getattr(event, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta is not None else None
        if text:
            yield text


def _without_reasoning(chunks: Iterable[str]) -> Iterator[str]:
    """Drop a leading ``<think>…</think>`` block from a token stream, even when tags split."""
    pending = ""
    state = "start"  # start -> thinking -> after_thinking -> text
    for chunk in chunks:
        if state == "text":
            yield chunk
            continue
        if state == "after_thinking":
            # Whitespace between </think> and the answer can arrive in later chunks.
            chunk = chunk.lstrip()
            if chunk:
                state = "text"
                yield chunk
            continue
        pending += chunk
        if state == "start":
            head = pending.lstrip()
            if not head or (_OPEN_THINK.startswith(head) and head != _OPEN_THINK):
                continue
            if head.startswith(_OPEN_THINK):
                state = "thinking"
                pending = head[len(_OPEN_THINK) :]
            else:
                state = "text"
                yield pending
                pending = ""
                continue
        if state == "thinking":
            end = pending.find(_CLOSE_THINK)
            if end == -1:
                pending = pending[-len(_CLOSE_THINK) :]
                continue
            rest = pending[end + len(_CLOSE_THINK) :].lstrip()
            pending = ""
            if rest:
                state = "text"
                yield rest
            else:
                state = "after_thinking"
    if state == "start" and pending.strip():
        yield pending
