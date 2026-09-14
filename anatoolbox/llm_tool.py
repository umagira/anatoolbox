from __future__ import annotations

import json
from typing import Any

from anatoolbox.base import ToolContext
from anatoolbox.llm_client import call_llm_json, model_for


class LLMTool:
    """Base for tools whose work is a single structured-JSON LLM call —
    system_prompt + model + a user prompt built from args, parsed as JSON.
    Not related to LLMProvider/agent_loop's model turns at all — this is
    tool-internal; the orchestrating model never knows this happened, it only
    ever sees this tool's own returned result.

    Dual-use API:
    - ``run`` — LLM call + ``_build_result`` → structured dict (pipelines).
    - ``execute`` — ``present(run(...))`` → JSON string for the agent.
    - ``present`` — default ``json.dumps``; override to add a render envelope.

    Subclasses still declare their own `schema` (input_schema genuinely
    differs per tool — nothing to derive it from), set `system_prompt` and
    `model` or `role` (see llm_client.configure_llm), and implement
    `_build_user_prompt`. Override `_build_result` only if the tool needs to
    reshape the LLM's parsed JSON (e.g. keeping the original args alongside
    the extracted fields) — the default returns it as-is. Do **not** embed a
    ``render`` envelope in ``_build_result``; override ``present`` instead.
    Override `_select_model` to vary the model per call (e.g. by input
    complexity) instead of using the fixed `model` attribute — the default
    just returns `self.model` unchanged.

    Set `response_schema` (a JSON Schema dict) to constrain the LLM's output
    to that exact shape via strict structured output where the endpoint
    supports it (degrading to JSON mode, then prompt-only), rather than the
    default loose `json_object` mode where the shape is only as reliable as
    the system prompt's prose description of it — see call_llm_json for the
    strict-mode schema constraints this requires.
    """

    system_prompt: str = ""
    #: Explicit model name. None means "the model the configured role maps to".
    model: str | None = None
    #: Model role this tool asks for; see ``llm_client.configure_llm``.
    role: str = "default"
    temperature: float = 0.0
    response_schema: dict[str, Any] | None = None

    def _build_user_prompt(self, args: dict[str, Any]) -> str:
        raise NotImplementedError

    def _select_model(self, args: dict[str, Any]) -> str:
        return self.model or model_for(self.role)

    def _system_prompt_for(self, args: dict[str, Any], *, context: ToolContext) -> str:
        """Override to vary the system prompt by project/context."""
        return self.system_prompt

    def _build_result(self, args: dict[str, Any], llm_output: dict[str, Any]) -> dict[str, Any]:
        return llm_output

    def present(self, data: dict[str, Any]) -> str:
        """Wrap ``run`` data for the agent. Default: plain JSON (no render)."""
        return json.dumps(data)

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        llm_output = call_llm_json(
            self._system_prompt_for(args, context=context),
            self._build_user_prompt(args),
            model=self._select_model(args),
            temperature=self.temperature,
            response_schema=self.response_schema,
            project=context.project,
        )
        return self._build_result(args, llm_output)

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str:
        return self.present(self.run(args, context=context))
