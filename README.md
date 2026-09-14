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
| **Gather** | `ingest_` `retrieve_` `rerank_` |
| **Preprocess** | `parse_` `clean_` `normalize_` `deduplicate_` `chunk_` |
| **Extract** | `extract_` `detect_` `resolve_` |
| **Analyze** | `aggregate_` `calculate_` `classify_` `compare_` `score_` |
| **Enrich** | `synthesize_` `enrich_` `benchmark_` `interpret_` `explain_` `assess_` `recommend_` `validate_` |

Presentation — charts, tables, reports — is deliberately **not** a stage. It
belongs in the frontend: a tool can declare how its result may be displayed
(`render_type`), and the host application renders it.

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

## Run it on a file

No services, no cluster, no index. Point it at a CSV:

```python
import anatoolbox
from anatoolbox import ToolContext, resolve_tools

anatoolbox.register_reference_tools()
ingest, chunk, retrieve, rerank = resolve_tools(
    ["ingest_corpus", "chunk_articles_by_paragraph", "retrieve_passages", "rerank_passages"]
)

ingest.run({"path": "ai_media.csv", "text_field": "content"}, context=ctx)
# -> corpus_1

chunk.run({"target_words": 150}, context=ctx)
# -> corpus_2: paragraph chunks, each linked to its article, derived_from ['corpus_1']

retrieve.run({"query": "agentic web standards", "strategy": "hybrid", "size": 30}, context=ctx)
# -> passages_1: a generous candidate set, derived_from ['corpus_2']

rerank.run({"keep": 5}, context=ctx)
# -> passages_2: the candidates re-scored on their full text, derived_from ['passages_1']
```

`strategy` is `sparse` (BM25), `dense` (embeddings), or `hybrid` (reciprocal
rank fusion of the two) — so comparing retrieval strategies is a changed
argument, not a changed pipeline.

## Try-out notebooks

Each build day ships a notebook that exercises that day's features on the real
[AI Media Dataset](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset),
framed around the HSLU *Computational Language Technologies* capstone. The dataset is
downloaded from Kaggle's public API on first run — no Kaggle account needed.

| Notebook | Covers |
|---|---|
| [`notebooks/day2_retrieval_ai_media.ipynb`](notebooks/day2_retrieval_ai_media.ipynb) | Ingesting the corpus, sparse/dense/hybrid retrieval, date windows, chunking, lineage |

## Install

```bash
pip install anatoolbox                  # contracts + memory + BM25 retrieval
pip install "anatoolbox[embeddings]"    # + dense retrieval with real models
pip install "anatoolbox[local]"         # + parquet corpora
```

The core has a single third-party dependency. BM25 is implemented in-tree, and
dense ranking falls back to plain Python when numpy is absent, so sparse and
dense retrieval both work on a bare install — the extras buy speed and real
embedding models, not basic functionality.

## Status

| Area | State |
|---|---|
| Prefix contracts (25) | ✅ complete |
| Recordset memory | ✅ complete, pluggable store + policy |
| Registry / plugin entry points | ✅ complete |
| Local corpus backend (CSV/JSON/JSONL/Parquet) | ✅ complete |
| Retrieval: BM25 / dense / hybrid | ✅ complete |
| Reference instantiations | 🚧 `ingest_corpus`, `chunk_articles_by_paragraph`, `retrieve_passages`, `rerank_passages` |
| Answering + evaluation tools | ❌ not yet |
| Docs site | ❌ not yet |

## License

Apache-2.0. See [LICENSE](LICENSE).
