"""CLI: run any registered toolbox tool from a JSON fixture (or inline args).

Examples::

    python -m anatoolbox.run_tool --fixture synthesize_competitor_summary.cabin_ifec --dry-run
    python -m anatoolbox.run_tool --fixture synthesize_competitor_summary.cabin_ifec
    python -m anatoolbox.run_tool --fixture path/to/fixture.json --out /tmp/out.json
    python -m anatoolbox.run_tool --tool extract_query_terms --args '{"query":"SAF"}'
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from anatoolbox.base import Tool, ToolContext
from anatoolbox.errors import ToolInputError
from anatoolbox.llm_client import configure_openai_key_resolver
from anatoolbox.registry import resolve_tools
from anatoolbox.tool_fixtures import (
    FIXTURES_DIR,
    build_tool_context,
    capture_recordsets,
    dry_run_tool,
    load_fixture,
    resolve_fixture_path,
    run_gather_steps,
)


def _parse_args_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("{"):
        data = json.loads(text)
    else:
        path = Path(text).expanduser()
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--args must be a JSON object or a file containing one")
    return data


def _write_out(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    if path.suffix.lower() == ".md" and isinstance(payload, dict) and "content" in payload:
        path.write_text(str(payload.get("content") or ""), encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _supports_streaming_execute(tool: Tool) -> bool:
    execute = getattr(tool, "execute", None)
    return callable(execute) and inspect.isgeneratorfunction(execute)


def stream_execute(
    tool: Tool,
    args: dict[str, Any],
    *,
    context: ToolContext,
    stream_file=None,
) -> tuple[str, str]:
    """Drive a generator ``execute``; print token deltas as they arrive.

    Returns ``(final_result, streamed_text)``. ``final_result`` is the value
    returned by the generator (often a JSON render string); ``streamed_text``
    is the concatenation of ``ToolProgress.text`` chunks.
    """
    out = stream_file if stream_file is not None else sys.stdout
    generator = tool.execute(args, context=context)
    parts: list[str] = []
    try:
        while True:
            progress = next(generator)
            if progress.text:
                parts.append(progress.text)
                out.write(progress.text)
                out.flush()
            elif progress.tool_name and progress.result is None:
                print(
                    f"\n[{progress.tool_name} starting]\n",
                    file=sys.stderr,
                    flush=True,
                )
            elif progress.tool_name and progress.result is not None:
                print(
                    f"\n[{progress.tool_name} done]\n",
                    file=sys.stderr,
                    flush=True,
                )
    except StopIteration as stop:
        final = stop.value if stop.value is not None else "".join(parts)
        if not str(final).endswith("\n"):
            out.write("\n")
            out.flush()
        return str(final), "".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m anatoolbox.run_tool",
        description=(
            "Run a toolbox tool with fixture or inline args, bypassing the "
            "research brief / workflow. Default --mode execute (agent path; "
            "streams when execute() is a generator). Use --mode run for the "
            "blocking pipeline dict. --dry-run inspects prompts without LLM."
        ),
    )
    parser.add_argument(
        "--fixture",
        help=f"Fixture name under {FIXTURES_DIR} or a path to a JSON fixture",
    )
    parser.add_argument("--tool", help="Tool name when not using --fixture")
    parser.add_argument(
        "--args",
        help="JSON object string, or path to a JSON object file (with --tool)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Override project slug (default: the fixture's project, else demo)",
    )
    parser.add_argument(
        "--registry",
        default="default",
        help="Tool registry name (default: default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs / build prompts; do not call the LLM",
    )
    parser.add_argument(
        "--mode",
        choices=("execute", "run"),
        default="execute",
        help=(
            "execute (default): agent path — stream token deltas when "
            "execute() is a generator, else print present()/execute JSON. "
            "run: pipeline path — blocking tool.run() → one JSON dict. "
            "For streaming tools these are not the same code path."
        ),
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help=argparse.SUPPRESS,  # alias for --mode run
    )
    parser.add_argument(
        "--no-gather",
        action="store_true",
        help=(
            "Skip the fixture's gather steps (no Elasticsearch); use only "
            "inline args / seeded recordsets"
        ),
    )
    parser.add_argument(
        "--capture",
        help=(
            "After gather, write a replayable offline fixture (real records inlined) to this path"
        ),
    )
    parser.add_argument(
        "--out",
        help="Write final result JSON/markdown to this path",
    )
    parser.add_argument(
        "--session-id",
        default="fixture",
        help="ToolContext session_id (default: fixture)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override OpenAI key (default: OPENAI_API_KEY_<PROJECT> from env/.env)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    opts = parser.parse_args(argv)

    recordset_entries: list[dict[str, Any]] | None = None
    gather_steps: list[dict[str, Any]] | None = None
    if opts.fixture:
        fixture_path = resolve_fixture_path(opts.fixture)
        fixture = load_fixture(fixture_path)
        tool_name = str(opts.tool or fixture.get("tool") or "").strip()
        args = dict(fixture.get("args") or {})
        if opts.args:
            args.update(_parse_args_json(opts.args))
        project = str(opts.project or fixture.get("project") or "demo")
        raw_rs = fixture.get("recordsets")
        if raw_rs is not None:
            if not isinstance(raw_rs, list):
                parser.error("fixture.recordsets must be a list")
            recordset_entries = raw_rs
        raw_gather = fixture.get("gather")
        if raw_gather is not None:
            if not isinstance(raw_gather, list):
                parser.error("fixture.gather must be a list")
            gather_steps = None if opts.no_gather else raw_gather
    else:
        if not opts.tool:
            parser.error("Provide --fixture or --tool")
        tool_name = opts.tool.strip()
        args = _parse_args_json(opts.args) if opts.args else {}
        project = str(opts.project or "demo")

    if not tool_name:
        parser.error("Tool name missing (fixture.tool or --tool)")

    try:
        tool = resolve_tools([tool_name], registry=opts.registry)[0]
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Same per-project keys as the agent (OPENAI_API_KEY_MYPROJECT, …).
    if opts.api_key:
        configure_openai_key_resolver(lambda _project: str(opts.api_key).strip())
    else:
        # Per-project key if set (OPENAI_API_KEY_MYPROJECT, …), else the
        # ordinary OPENAI_API_KEY. Hosts that need richer scoping register
        # their own resolver before calling in.
        import os

        def _resolve(project: str) -> str:
            scoped = f"OPENAI_API_KEY_{project.strip().upper()}"
            key = os.environ.get(scoped) or os.environ.get("OPENAI_API_KEY") or ""
            if not key:
                raise RuntimeError(
                    f"No API key: set {scoped} or OPENAI_API_KEY, or pass --api-key."
                )
            return key

        configure_openai_key_resolver(_resolve)

    context, handles = build_tool_context(
        project=project,
        session_id=opts.session_id,
        recordset_entries=recordset_entries,
    )

    if gather_steps:
        # Real evidence means a real source system. Only fixtures with a
        # `gather` block need one, so the dependency stays optional.
        try:
            from anatoolbox.contrib.elasticsearch import configure_es
        except ImportError:
            print(
                "error: this fixture declares `gather` steps, which need a source "
                "system. Install the backend extra, or use a fixture with canned "
                "`recordsets`, or pass --no-gather.",
                file=sys.stderr,
            )
            return 2
        configure_es()
        try:
            for entry in run_gather_steps(gather_steps, context=context, registry=opts.registry):
                counts = ", ".join(
                    f"{handle} ({count})" for handle, count in entry["counts"].items()
                )
                print(
                    f"[gather] {entry['tool']} -> {counts or 'no records'}",
                    file=sys.stderr,
                    flush=True,
                )
                handles.extend(entry["handles"])
        except Exception as exc:  # noqa: BLE001 — CLI surface, not a library
            print(f"error: gather step failed: {exc}", file=sys.stderr)
            return 2

    if opts.capture:
        captured = {
            "project": project,
            "tool": tool_name,
            "args": args,
            "recordsets": capture_recordsets(context.recordsets),
        }
        _write_out(Path(opts.capture).expanduser(), captured)
        print(f"captured replayable fixture -> {opts.capture}", file=sys.stderr, flush=True)

    if opts.dry_run:
        try:
            result: Any = dry_run_tool(tool, args, context=context)
        except ToolInputError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if handles:
            result["seeded_handles"] = handles
        print(json.dumps(result, indent=2, default=str))
        if opts.out:
            _write_out(Path(opts.out).expanduser(), result)
            print(f"wrote {opts.out}", file=sys.stderr)
        return 0

    mode = "run" if opts.no_stream else opts.mode
    use_stream = mode == "execute" and _supports_streaming_execute(tool)
    print(f"mode={mode}" + (" (streaming)" if use_stream else ""), file=sys.stderr, flush=True)

    try:
        if mode == "execute" and use_stream:
            final, streamed = stream_execute(tool, args, context=context)
            if opts.out:
                out_path = Path(opts.out).expanduser()
                if out_path.suffix.lower() == ".md" and streamed:
                    _write_out(out_path, streamed)
                else:
                    _write_out(out_path, final)
                print(f"\nwrote {opts.out}", file=sys.stderr)
            return 0

        if mode == "execute":
            # Non-generator execute: same as agent path (often present(run(...))).
            execute = getattr(tool, "execute", None)
            if not callable(execute):
                print(f"error: tool {tool_name!r} has no execute()", file=sys.stderr)
                return 2
            final = execute(args, context=context)
            print(final if isinstance(final, str) else json.dumps(final, indent=2, default=str))
            if opts.out:
                _write_out(Path(opts.out).expanduser(), final)
                print(f"wrote {opts.out}", file=sys.stderr)
            return 0

        run = getattr(tool, "run", None)
        if not callable(run):
            print(f"error: tool {tool_name!r} has no run()", file=sys.stderr)
            return 2
        result = run(args, context=context)
        if isinstance(result, dict) and handles:
            result = {**result, "seeded_handles": handles}
        print(json.dumps(result, indent=2, default=str))
        if opts.out:
            _write_out(Path(opts.out).expanduser(), result)
            print(f"wrote {opts.out}", file=sys.stderr)
        return 0
    except (RuntimeError, ToolInputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
