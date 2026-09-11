from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

MODELS = {
    "cheap": "gpt-5-nano",
    "cheap_nonreasoning": "gpt-4.1-nano",
    "medium": "gpt-5-mini",
    "expensive": "gpt-5.1",
    "synthesis": "gpt-5.6-luna",
}

# Per-project OpenAI clients (keyed by project slug).
_clients: dict[str, OpenAI] = {}

# Set by the agent at startup (see app.create_app). Pipelines/notebooks can
# call configure_openai_key_resolver themselves or pass api_key= explicitly.
_openai_key_resolver: Callable[[str], str] | None = None


def configure_openai_key_resolver(resolver: Callable[[str], str]) -> None:
    """Register how tool-internal OpenAI calls obtain a per-project API key."""
    global _openai_key_resolver
    _openai_key_resolver = resolver
    _clients.clear()


def clear_openai_clients() -> None:
    """Test helper."""
    _clients.clear()


def _resolve_api_key(*, project: str | None, api_key: str | None) -> str:
    if api_key is not None and str(api_key).strip():
        return str(api_key).strip()
    if not project or not str(project).strip():
        raise RuntimeError(
            "call_llm_json requires project= (or api_key=). "
            "Per-project OpenAI keys are mandatory — there is no global fallback."
        )
    if _openai_key_resolver is None:
        raise RuntimeError(
            "OpenAI key resolver is not configured. "
            "The agent API calls configure_openai_key_resolver() at startup; "
            "pipelines should pass api_key= or configure a resolver."
        )
    return _openai_key_resolver(str(project).strip())


def _get_client(*, project: str | None = None, api_key: str | None = None) -> OpenAI:
    key = _resolve_api_key(project=project, api_key=api_key)
    # Cache by the secret itself so distinct projects with distinct keys
    # don't share a client; tests that pass api_key="test" share one entry.
    cache_key = key
    client = _clients.get(cache_key)
    if client is None:
        client = OpenAI(api_key=key)
        _clients[cache_key] = client
    return client


def _model_accepts_temperature(model: str) -> bool:
    """Some OpenAI models only allow the default temperature (omit the param)."""
    name = (model or "").strip().lower()
    # gpt-5* and reasoning models reject temperature=0; only default (1) works.
    if (
        name.startswith("gpt-5")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    ):
        return False
    return True


def _default_reasoning_effort(model: str) -> str | None:
    """gpt-5-nano defaults to medium reasoning, which dominates brief latency.

    Structured JSON extraction does not need that chain-of-thought; ``low``
    keeps world knowledge (company cohorts) without the extra thinking tokens.
    Larger gpt-5 models keep the API default unless the caller overrides.
    """
    name = (model or "").strip().lower()
    if name == "gpt-5-nano" or name.startswith("gpt-5-nano-"):
        return "low"
    return None


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float | None = 0.0,
    response_schema: dict[str, Any] | None = None,
    project: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Generic structured-JSON LLM call, shared by every LLMTool subclass.

    Not part of the LLMProvider/agent_loop abstraction — that's for the
    top-level streaming, tool-calling model turns run_turn() drives. This is
    a plain, synchronous, non-streaming, no-tool-calling request, entirely
    tool-internal: the orchestrating model never knows this call happened,
    it only ever sees this tool's own returned result.

    Authentication is per-project: pass ``project=`` (resolved via
    ``configure_openai_key_resolver``) or an explicit ``api_key=``. There is
    no fallback to a process-wide ``OPENAI_API_KEY``.

    `response_schema`, when given, switches from loose `json_object` mode
    to OpenAI's Structured Outputs (`json_schema` + `strict: true`).

    ``temperature`` is omitted for models that only accept the API default
    (e.g. gpt-5-mini), and whenever the caller passes ``temperature=None``.

    ``reasoning_effort`` is passed through when set. ``gpt-5-nano`` defaults
    to ``low`` so structured JSON calls do not sit on medium reasoning.
    Pass ``reasoning_effort=""`` to omit the parameter entirely.
    """
    response_format: dict[str, Any] = (
        {"type": "json_object"}
        if response_schema is None
        else {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": response_schema},
        }
    )
    create_kwargs: dict[str, Any] = {
        "model": model,
        "response_format": response_format,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if temperature is not None and _model_accepts_temperature(model):
        create_kwargs["temperature"] = temperature
    effort = reasoning_effort if reasoning_effort is not None else _default_reasoning_effort(model)
    if effort:
        create_kwargs["reasoning_effort"] = effort
    response = _get_client(project=project, api_key=api_key).chat.completions.create(
        **create_kwargs
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty response")
    return json.loads(content)


def call_llm_text_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float | None = 0.0,
    project: str | None = None,
    api_key: str | None = None,
):
    """Stream plain-text completion chunks (tool-internal; not the agent provider).

    Yields non-empty ``delta.content`` strings as they arrive. Callers that need
    the full string can ``"".join(call_llm_text_stream(...))``.
    """
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }
    if temperature is not None and _model_accepts_temperature(model):
        create_kwargs["temperature"] = temperature
    stream = _get_client(project=project, api_key=api_key).chat.completions.create(**create_kwargs)
    for event in stream:
        choices = getattr(event, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta is not None else None
        if text:
            yield text


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float | None = 0.0,
    project: str | None = None,
    api_key: str | None = None,
) -> str:
    """Non-streaming plain-text completion (collects ``call_llm_text_stream``)."""
    content = "".join(
        call_llm_text_stream(
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
            project=project,
            api_key=api_key,
        )
    )
    if not content.strip():
        raise ValueError("LLM returned empty response")
    return content
