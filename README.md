# anatoolbox

**A task-type framework for research and analytics agents.**

Most agent frameworks give you a way to call tools. `anatoolbox` gives you a
way to say *what kind of analytical work a step is* — and a way to move
evidence between steps without pushing it through the model's context window.

> ⚠️ **Alpha.** The contracts are stable; the reference instantiations are not.
> Published initially as a research demo. Expect breaking changes before 1.0.

It was extracted from an internal analytics toolbox. Its first users are student teams in the
HSLU *Computational Language Technologies* capstone, who build a retrieval-augmented question
answering system over a public news dataset — which is what
[`notebooks/course_project_skeleton.ipynb`](notebooks/course_project_skeleton.ipynb) walks
through. Nothing in the package itself is specific to that course.

## Why this exists

Two ideas carry the package.

### 1. A grammar of analytical work

Research is not an undifferentiated pile of "tools". It is six stages, and
within them a small, closed set of **task types**, each named by a prefix:

| Stage | Task types |
|---|---|
| **Prepare** | `plan_` |
| **Gather** | `rewrite_` `ingest_` `retrieve_` `rerank_` |
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
`retrieve_passages`, `score_rag_answer`.

This is not decoration. An agent that knows a step is a `validate_` step knows
what it may consume and what it must produce, without being told about
specific functions. Registration enforces it: a tool whose name does not start
with a known prefix is rejected.

### 2. Recordset dataflow

In an agent, stage chaining runs through **handles**, not through the conversation.

```python
handle = context.recordsets.remember(
    object_type="passages",
    stage="gather",
    produced_by="retrieve_passages",
    args={"query": "agentic web"},
    ref=value_ref(rows),
).handle
# -> "passages_1"

record = context.recordsets.bind(object_type="passages", tool_name="score_rag_answer")
rows = context.recordsets.records(record)  # fetched by reference, on demand
```

Every recordset carries provenance (`produced_by`, `args`) and lineage
(`derived_from`), so the evidence behind a score or a chart is exact rather
than reconstructed. Document bodies never enter recordset metadata, the prompt
digest, or working memory — only pointers do.

## The shape of a tool

Every tool is dual-use: `run` returns data for pipelines and notebooks, `execute` returns a
JSON string for an agent. Subclass `BaseTool` and it builds the schema, argument checks,
provenance and `execute` from a few class attributes, so a tool is mostly its `run`:

```python
from collections import Counter

from anatoolbox import register_tool
from anatoolbox.analyze.aggregate.base import PREFIX, STAGE
from anatoolbox.provenance import run_id_of
from anatoolbox.tool import BaseTool


class AggregateArticlesByMonthTool(BaseTool):
    tool_name = "aggregate_articles_by_month"
    prefix, stage = PREFIX, STAGE
    description = "Count retrieved articles per publication month."
    input_schema = {
        "type": "object",
        "properties": {"input": {"type": "object", "description": "A retrieve_passages result."}},
        "required": ["input"],
    }

    def run(self, args, *, context):
        retrieved = args["input"]
        months = Counter(str(p.get("date", ""))[:7] for p in retrieved["passages"])
        return {
            "months": dict(sorted(months.items())),
            "provenance": self.provenance({}, derived_from=[run_id_of(retrieved)]),
        }


register_tool(AggregateArticlesByMonthTool())
```

`anatoolbox` ships **contracts, not a tool catalog** — you instantiate prefixes for your own
corpus. The tools that do ship are baselines.

## Change a shipped tool

Each shipped tool keeps its plumbing in `run()` and exposes the step a variant changes as a
hook method. Subclass, rename, override one hook — input binding, pipeline chaining and
provenance are inherited:

```python
import re
from anatoolbox.preprocess.chunk.chunk_by_size import ChunkBySizeTool


class ChunkBySentence(ChunkBySizeTool):
    tool_name = "chunk_by_sentence"

    def split(self, text, settings):
        return [{"text": s} for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
```

[`docs/extending.md`](docs/extending.md) lists every hook, how to add arguments, and when to
plug in a function or write a new tool instead.

## Run it on a file

No services, no cluster, no index. Point it at a CSV:

```python
import anatoolbox
from anatoolbox import ToolContext, resolve_tools

anatoolbox.register_reference_tools()
ingest, chunk, rewrite, retrieve, rerank, synthesize = resolve_tools(
    ["ingest_corpus", "chunk_by_size", "rewrite_query_for_retrieval",
     "retrieve_passages", "rerank_passages", "synthesize_answer"]
)
ctx = ToolContext()  # a notebook or batch pipeline: no memory needed
question = "Which standards for AI agents emerged, and who backs them?"

articles = ingest.run({"path": "ai_media.csv", "text_field": "content"}, context=ctx)
chunks = chunk.run({"input": articles, "size": 200, "overlap": 40}, context=ctx)
queries = rewrite.run({"question": question, "strategy": "decompose"}, context=ctx)
candidates = retrieve.run(
    {"query": question, "input": chunks, "queries_input": queries, "strategy": "hybrid", "size": 30},
    context=ctx,
)
top = rerank.run({"input": candidates, "keep": 6}, context=ctx)
answer = synthesize.run({"question": question, "input": top}, context=ctx)

answer["answer_markdown"]            # cited [S1]…[Sn], with invented labels, unused sources
                                     # and citation coverage reported alongside
answer["provenance"]["derived_from"]  # -> [top["provenance"]["run_id"]]
```

`strategy` is `sparse` (BM25), `dense` (embeddings), or `hybrid` (reciprocal
rank fusion of the two) — so comparing retrieval strategies is a changed
argument, not a changed pipeline.

## Evaluate it

```python
draft, metrics, judge = resolve_tools(
    ["extract_test_questions", "calculate_retrieval_metrics", "score_rag_answer"]
)

test_set = draft.run({"input": chunks, "sample_size": 100, "questions_per_record": 2}, context=ctx)
# review the questions by hand before using them

runs = [
    {"question": q["question"], "relevant": q["source_ids"],
     "retrieved": retrieve.run({"query": q["question"], "input": chunks}, context=ctx)}
    for q in test_set["questions"]
]
metrics.run({"results": runs, "k": [1, 5, 10]}, context=ctx)["summary"]   # precision, recall, hit, MRR

judge.run({"answer": answer, "reference_answer": test_set["questions"][0]["reference_answer"]}, context=ctx)
# -> faithfulness, relevance, context and temporal grounding, correctness; unsupported claims; review flags
```

Questions carry a type (factual, temporal, relational, analytical, comparative), so results
can be compared per category. Metrics count chunks as their article, once.

## Knowledge graphs

`anatoolbox` does not build knowledge graphs — it loads the one you built. An edge table or
NetworkX node-link JSON, one row per fact (`subject`, `relation`, `object`, optional types,
`source_ids` and `date`), becomes a `KnowledgeGraph` with plain lookups:

```python
from anatoolbox.graph import get_graph

loaded = resolve_tools(["ingest_knowledge_graph"])[0].run({"path": "graph_edges.csv"}, context=ctx)
graph = get_graph(loaded["graph"])
facts = graph.subgraph(graph.find_entities(question), hops=1, max_facts=15)
answer = synthesize.run({"question": question, "input": top, "facts": facts}, context=ctx)
# facts are cited as [G1]…[Gn] and checked like sources
```

## Pipelines and agents

The same tools run in two modes.

- **Pipelines and notebooks** pass each result into the next tool as `input`. Nothing is
  bound implicitly, and every result carries a `provenance` block: its `run_id`, the tool,
  the effective settings (defaults included), the package version, a timestamp, and the
  `run_id`s it was `derived_from`. It is plain JSON — save it next to your results, and you
  can later tell which corpus, chunking, queries, reranker and model produced each answer.
- **Agents** attach recordset memory: `ToolContext(recordsets=Memory(...))`. A language model
  then refers to results by short handles such as `passages_2` instead of copying data through
  its context window, and a tool called without `input` binds the newest compatible result.
  Provenance names those handles.

## Choose a language model

Tools that need a language model use any OpenAI-compatible endpoint — OpenAI, a
hosted provider, or a model on your own machine:

```python
from anatoolbox.llm_client import configure_llm

configure_llm(model="gpt-4o-mini")                                     # OpenAI; key from OPENAI_API_KEY
configure_llm(base_url="http://localhost:11434/v1", model="qwen3:4b")  # Ollama on your machine
configure_llm(models={"fast": "qwen3:1.7b", "strong": "qwen3:14b"})   # different models per role
```

Or set `ANATOOLBOX_LLM_BASE_URL`, `ANATOOLBOX_LLM_API_KEY` and `ANATOOLBOX_LLM_MODEL`.
There is no default model: which model produced an answer is part of the answer.
Query rewriting uses the `fast` role, answers the `strong` role, and test questions and
judging the `evaluation` role — set it to a different model than the one that answers.
Roles that are not configured fall back to the default model.
Endpoints differ in what they accept, so parameters an endpoint rejects are dropped
and remembered, and JSON requests fall back from strict schemas to JSON mode to
prompt-only.

## Notebooks

The notebook runs on the real
[AI Media Dataset](https://www.kaggle.com/datasets/jannalipenkova/ai-media-dataset),
framed around the HSLU *Computational Language Technologies* capstone. The dataset is
downloaded from Kaggle's public API on first run — no Kaggle account needed.

It is committed without outputs — run it yourself.

| Notebook | Covers |
|---|---|
| [`notebooks/course_project_skeleton.ipynb`](notebooks/course_project_skeleton.ipynb) | **The Stage 3 project skeleton**, structured like Chapter 7: search (build, evaluate, optimize) and RAG (set up, evaluate, optimize — incl. the Stage 1 knowledge graph), with a baseline and a measurement at every step, one worked optimization, and subclass templates for the rest |

## Install

Not on PyPI yet — install from the repository:

```bash
REPO=git+https://github.com/umagira/anatoolbox
pip install "anatoolbox @ $REPO"                # contracts + memory + BM25 retrieval
pip install "anatoolbox[embeddings] @ $REPO"    # + dense retrieval with real models
pip install "anatoolbox[parquet] @ $REPO"       # + parquet corpora
pip install "anatoolbox[local] @ $REPO"         # both of the above
```

The core has a single third-party dependency. BM25 is implemented in-tree, and
dense ranking falls back to plain Python when numpy is absent, so sparse and
dense retrieval both work on a bare install — the extras buy speed and real
embedding models, not basic functionality.

## Status

| Area | State |
|---|---|
| Prefix contracts (26) | ✅ complete |
| Recordset memory | ✅ complete, pluggable store + policy |
| Registry / plugin entry points | ✅ complete |
| Local corpus backend (CSV/JSON/JSONL/Parquet) | ✅ complete |
| Retrieval: BM25 / dense / hybrid | ✅ complete |
| Extending shipped tools by subclassing | ✅ `BaseTool` + hooks, see [`docs/extending.md`](docs/extending.md) |
| Reference instantiations | 🚧 `ingest_corpus`, `ingest_knowledge_graph`, `chunk_by_size`, `rewrite_query_for_retrieval`, `retrieve_passages`, `rerank_passages`, `synthesize_answer` |
| Any OpenAI-compatible LLM endpoint | ✅ complete |
| Answer synthesis with checked citations, incl. graph facts | ✅ `synthesize_answer` |
| Evaluation | ✅ `extract_test_questions`, `calculate_retrieval_metrics`, `score_rag_answer` |
| Knowledge graph loading and lookups | ✅ `anatoolbox.graph` |
| Docs site | ❌ not yet |

## License

Apache-2.0. See [LICENSE](LICENSE).
