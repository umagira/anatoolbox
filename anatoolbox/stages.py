"""Workflow stages of the analytical toolbox.

Six stages, each a package of task-type prefixes::

    anatoolbox/<stage>/<prefix>/base.py               # prefix contract + abstract I/O
    anatoolbox/<stage>/<prefix>/<prefix>_<object>.py  # instantiations

Presentation — charts, tables, reports — is deliberately not a stage: it
belongs in the frontend. A tool can declare how its result *may* be displayed
(``ToolSchema.render_type`` and the ``render`` envelope); the host renders it.
"""

STAGE_PACKAGE_BY_LABEL = {
    "Prepare": "prepare",
    "Gather": "gather",
    "Preprocess": "preprocess",
    "Extract information": "extract",
    "Analyze": "analyze",
    "Enrich and interpret": "enrich",
}

#: Stage order. Prepare scopes the work before any evidence is touched.
STAGE_ORDER = [
    "prepare",
    "gather",
    "preprocess",
    "extract",
    "analyze",
    "enrich",
]

#: What each stage is for — the job a prefix must fit to belong to it.
STAGE_DESCRIPTIONS = {
    "prepare": "Scope and plan the research before any evidence work: turn a request into a reviewable brief.",
    "gather": "Acquire source records — rewrite the question into queries, ingest, retrieve, and rerank — without changing what the records say.",
    "preprocess": "Prepare data for further analysis: parse, clean, normalize, deduplicate, and chunk it.",
    "extract": "Take raw data or information and extract the relevant subset in a given form: facts, entities, events, relationships.",
    "analyze": "Derive results by explicit methods: aggregates, calculations, classifications, comparisons, and scores.",
    "enrich": "Add context and judgment: synthesize evidence, interpret and explain results, assess, benchmark, validate, and recommend.",
}

#: Stages an evidence pipeline runs once the plan is approved. Prepare
#: produces the plan itself rather than running as a step of it.
RESEARCH_PIPELINE_STAGES = [stage for stage in STAGE_ORDER if stage != "prepare"]


# --- Prefix catalog -------------------------------------------------------
#
# Discovered from the shipped contracts (``<stage>/<prefix>/base.py``) rather
# than hardcoded, so adding a prefix package is the only step needed. New
# prefixes are rare; new instantiations of an existing prefix are the normal
# path (see the package README).

_PREFIX_TO_STAGE: "dict[str, str] | None" = None


def _discover_prefixes() -> "dict[str, str]":
    """Read PREFIX/STAGE out of each contract without importing it.

    Parsed with ``ast`` rather than matched with a regex: contracts are a mix
    of generated single-line assignments and hand-written parenthesized ones,
    and a regex silently drops whichever style it was not written for.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).parent
    found: dict[str, str] = {}
    for base in sorted(root.glob("*/*/base.py")):
        try:
            tree = ast.parse(base.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken contract is a build error
            continue
        values: dict[str, str] = {}
        for node in tree.body:
            target = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target.id, node.value
            elif (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                target, value = node.targets[0].id, node.value
            if target in ("PREFIX", "STAGE") and value is not None:
                try:
                    literal = ast.literal_eval(value)
                except ValueError:
                    continue
                if isinstance(literal, str):
                    values[target] = literal
        if "PREFIX" in values and "STAGE" in values:
            found[values["PREFIX"]] = values["STAGE"]
    return found


def prefix_to_stage() -> "dict[str, str]":
    """Map every shipped prefix (``'retrieve_'``) to its stage (``'gather'``)."""
    global _PREFIX_TO_STAGE
    if _PREFIX_TO_STAGE is None:
        _PREFIX_TO_STAGE = _discover_prefixes()
    return dict(_PREFIX_TO_STAGE)


def prefixes_for_stage(stage: str) -> "list[str]":
    """Every prefix belonging to ``stage``, sorted."""
    return sorted(p for p, s in prefix_to_stage().items() if s == stage)


def stage_for_tool_name(tool_name: str) -> "str | None":
    """Stage implied by a tool's name, or None if it matches no known prefix.

    ``aggregate_votes_by_topic`` -> ``'analyze'``. Longest prefix wins so a
    future ``extract_`` vs ``extract_foo_`` pair cannot collide.
    """
    matches = [p for p in prefix_to_stage() if tool_name.startswith(p)]
    if not matches:
        return None
    return prefix_to_stage()[max(matches, key=len)]
