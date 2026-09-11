# anatoolbox

**A task-type framework for research and analytics agents.**

Most agent frameworks give you a way to call tools. `anatoolbox` gives you a
way to say *what kind of analytical work a step is* — and a way to move
evidence between steps without pushing it through the model's context window.

> ⚠️ **Alpha.** The contracts are stable; the reference instantiations are not.
> Published initially as a research demo. Expect breaking changes before 1.0.

## Why this exists

Two ideas carry the package.

### 1. A grammar of analytical work

Research is not an undifferentiated pile of "tools". It is six stages, and
within them a small, closed set of **task types**, each named by a prefix:

| Stage | Task types |
|---|---|
| **Prepare** | `plan_` |
| **Gather** | `retrieve_` `ingest_` `monitor_` |
| **Extract** | `extract_` `parse_` `normalize_` `detect_` `resolve_` `deduplicate_` |
| **Analyze** | `calculate_` `aggregate_` `classify_` `identify_` `compare_` `trend_` `score_` `model_` |
| **Enrich** | `synthesize_` `enrich_` `benchmark_` `interpret_` `explain_` `assess_` `recommend_` `validate_` `weight_` `calibrate_` |
| **Present** | `chart_` `tabulate_` `visualize_` `summarize_` `report_` `export_` |

Each prefix ships a **contract**: abstract input, abstract output, a
one-sentence transformation, and a `Protocol`. A concrete tool is named
`<prefix><object>[_by_<dimension>][_for_<purpose>]` — `aggregate_mentions_by_period`,
`retrieve_passages`, `score_rag_answers`.

This is not decoration. An agent that knows a step is a `validate_` step knows
what it may consume and what it must produce, without being told about
specific functions. Registration enforces it: a tool whose name does not start
with a known prefix is rejected.

### 2. Recordset dataflow

Stage chaining runs through **handles**, not through the conversation.

```python
handle = context.recordsets.remember(
    object_type="passages",
    stage="gather",
    produced_by="retrieve_passages",
    args={"query": "agentic web"},
    ref=value_ref(rows),
).handle
# -> "passages_1"

record = context.recordsets.bind(object_type="passages", tool_name="score_rag_answers")
rows = context.recordsets.records(record)  # fetched by reference, on demand
```

Every recordset carries provenance (`produced_by`, `args`) and lineage
(`derived_from`), so the evidence behind a score or a chart is exact rather
than reconstructed. Document bodies never enter recordset metadata, the prompt
digest, or working memory — only pointers do.

## The shape of a tool

Every tool is dual-use. `run` is the pipeline/notebook API; `execute` is the
agent path.

```python
from anatoolbox import ToolSchema, ToolContext, register_tool, resolve_tools


class AggregateMentionsByPeriodTool:
    schema = ToolSchema(
        name="aggregate_mentions_by_period",
        description="Group mention counts into calendar periods.",
        input_schema={
            "type": "object",
            "properties": {"grain": {"type": "string"}},
            "required": ["grain"],
        },
    )

    def run(self, args, *, context) -> dict:
        return {"periods": [...]}

    def execute(self, args, *, context) -> str:
        import json

        return json.dumps(self.run(args, context=context))


register_tool(AggregateMentionsByPeriodTool())
(tool,) = resolve_tools(["aggregate_mentions_by_period"])
tool.run({"grain": "month"}, context=ToolContext(project="demo", session_id="s1"))
```

That is the whole extension story. `anatoolbox` ships **contracts, not a tool
catalog** — you instantiate prefixes for your own corpus.

## Install

```bash
pip install anatoolbox            # core: contracts + memory + LLM client
pip install "anatoolbox[local]"   # + run a corpus from a local file
```

The core has a single third-party dependency. Storage backends and embedding
stacks are extras, so `import anatoolbox` never pulls in a cluster driver.

## Status

| Area | State |
|---|---|
| Prefix contracts (34) | ✅ complete |
| Recordset memory | ✅ complete, pluggable store + policy |
| Registry / plugin entry points | ✅ complete |
| Reference instantiations | 🚧 in progress |
| Local corpus backend | 🚧 in progress |
| Docs site | ❌ not yet |

## License

Apache-2.0. See [LICENSE](LICENSE).
