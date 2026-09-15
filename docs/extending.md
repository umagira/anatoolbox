# Extending anatoolbox

The tools that ship with anatoolbox are **baselines**: standard, simple methods you can
measure your own ideas against. There are three ways to try something else, from least to
most code. Pick the smallest that fits.

| You want to… | Do this | Example |
|---|---|---|
| swap a model or a small function | **plug it in** | your fine-tuned embedding model |
| change one step of a shipped tool | **subclass it and override a hook** | sentence-based chunking, weighted hybrid search, a self-checking answer |
| do a kind of work no shipped tool does | **write a new tool** from a task-type contract | `extract_entities`, `aggregate_mentions_by_month` |

Whichever you choose, your variant runs in the same pipeline as the baseline, its results
carry provenance naming it, and a comparison table can tell the two apart.

## 1. Plug in a function

Some steps take a function you configure once:

```python
from anatoolbox.corpus import configure_embedder
from anatoolbox.reranking import configure_reranker
from anatoolbox.preprocess.chunk.chunk_by_size import register_contextualizer
from anatoolbox.llm_client import configure_llm

configure_embedder(my_model.encode)                  # texts -> vectors; dense and hybrid retrieval use it
configure_reranker(my_scorer, label="my-reranker")   # (query, texts) -> scores
register_contextualizer("title_and_date",            # then chunk_by_size(contextualize="title_and_date")
                        lambda record, chunk: f"{record['title']} ({record['date']})")
configure_llm(models={"strong": "…", "evaluation": "…"})  # a model per role
```

## 2. Subclass a tool and override a hook

Every shipped tool splits its work into a fixed `run()` — reading arguments, binding its
input, recording provenance, shaping the result — and small **hook methods** for the step a
variant changes. Override a hook and inherit everything else.

```python
import re
from anatoolbox.preprocess.chunk.chunk_by_size import ChunkBySizeTool


class ChunkBySentence(ChunkBySizeTool):
    tool_name = "chunk_by_sentence"            # 1. a new name, keeping the prefix
    description = "Split records into chunks of whole sentences, up to `size` words each."

    def split(self, text, settings):           # 2. override one hook
        chunks, current = [], []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if current and len(" ".join(current + [sentence]).split()) > settings["size"]:
                chunks.append({"text": " ".join(current)})
                current = []
            current.append(sentence)
        if current:
            chunks.append({"text": " ".join(current)})
        return chunks


chunks = ChunkBySentence().run({"input": articles, "size": 200}, context=ctx)   # 3. use it like the baseline
chunks["provenance"]["tool"]    # -> "chunk_by_sentence"
```

The three rules:

1. **Give the subclass its own `tool_name`**, starting with the same prefix (`chunk_`,
   `retrieve_`, …). Results, errors and provenance use it, and the registry refuses a
   second tool under an existing name, so a variant cannot silently replace the baseline.
2. **Override hooks, not `run()`.** `run()` is where inputs are bound and provenance is
   recorded; overriding it means redoing that.
3. **A new argument goes into `settings()` and `input_schema`.** `settings()` reads and
   checks arguments, and everything it returns is recorded in provenance — so a result
   says which settings produced it. `with_properties` extends the schema without changing
   the parent's:

```python
from anatoolbox.tool import with_properties


class ChunkBySentence(ChunkBySizeTool):
    tool_name = "chunk_by_sentence"
    input_schema = with_properties(ChunkBySizeTool.input_schema, {
        "max_sentences": {"type": "integer", "minimum": 1, "description": "Sentences per chunk."},
    })

    def settings(self, args):
        return {**super().settings(args),
                "max_sentences": self.int_arg(args, "max_sentences", 5)}
```

Call `super()` inside a hook to build on the baseline instead of replacing it — for example
`super().rank(...)` to re-order the baseline's hits, or `super().system_prompt(settings) + "…"`
to add an instruction.

### Hooks of the shipped tools

| Tool | Hooks | Typical variants |
|---|---|---|
| `ingest_corpus` | `load(settings)`, `prepare(corpus, settings)` | other file formats, filtering or cleaning records on the way in |
| `ingest_knowledge_graph` | `load(settings)`, `prepare(graph, settings)` | another graph format, keeping only your track's relation types, adding aliases |
| `chunk_by_size` | `split(text, settings)`, `contextualize(record, chunk_text, settings)`, `corpus_name(source, settings)` | sentence, paragraph, semantic or hierarchical chunking; LLM-written chunk context |
| `rewrite_query_for_retrieval` | `system_prompt(settings)`, `user_prompt(settings)`, `rewrite(settings, context)`, `clean(reply, settings)` | query expansion, transformation and classification; hypothetical answers to embed |
| `retrieve_passages` | `rank(query, corpus, settings, limit)`, `keep(record, settings)`, `boost(record, settings)`, `fuse(rankings, settings)` | weighted hybrid scoring, another index, metadata rules, entity-anchored retrieval |
| `rerank_passages` | `score(query, texts, settings)`, `reranker(settings)` | another cross-encoder, an LLM judging relevance |
| `synthesize_answer` | `curate(passages, settings)`, `curate_facts(facts, settings)`, `system_prompt(settings)`, `user_prompt(question, sources, settings, facts)`, `generate(system, user, settings, context)` | context fusion and compression, choosing graph facts, prompt constraints, chain-of-thought, reflection rounds |
| `extract_test_questions` | `sample(corpus, settings)`, `question_types` (class attribute), `system_prompt(settings)`, `user_prompt(record, text, settings)`, `generate(...)`, `clean(reply, record, settings)` | samples stratified by month or entity, your own question types |
| `calculate_retrieval_metrics` | `metrics(retrieved, relevant, settings)`, `retrieved_ids(item, settings)`, `relevant_ids(item, settings)` | nDCG, average precision |
| `score_rag_answer` | `criteria` (class attribute), `system_prompt(settings)`, `user_prompt(question, answer, sources, settings, facts)`, `judge(...)`, `clean(reply, settings)`, `review(judged, settings)` | your own criteria, several judges, stricter checks on the judge |

Knowledge graphs are not a tool but a data structure: `anatoolbox.graph.KnowledgeGraph` holds
your Stage 1 graph and offers plain lookups (`find_entities`, `facts_about`, `neighbors`,
`subgraph`, `fact_source_ids`). Graph-aware retrieval is built from those — typically a
`retrieve_passages` subclass whose `rank` or `boost` uses the graph, plus facts passed to
`synthesize_answer(facts=...)`.

Each tool's module docstring and hook docstrings say what a hook receives and must return.
In a notebook, `help(RetrievePassagesTool.rank)` shows one hook's documentation; the course
notebook's `show_hooks(RetrievePassagesTool)` lists all of a tool's hooks at once.

Every tool also has `settings(args)`. The helpers on the base class keep argument checks
short and their errors consistent: `self.text_arg`, `self.int_arg`, `self.choice_arg`, and
`self.input_error(message, ...)` for anything else.

## 3. Write a new tool from a task-type contract

When no shipped tool does the kind of work you need, write one. First find its **task type**:
the prefixes are listed in the README, and each has a contract in
`anatoolbox/<stage>/<prefix>/base.py` saying what such a tool consumes and produces.
Extracting entities is `extract_`; counting mentions per month is `aggregate_`.

Subclass `BaseTool` for the schema, argument helpers, provenance and `execute`:

```python
from collections import Counter

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
        retrieved = args.get("input")
        if not isinstance(retrieved, dict) or "passages" not in retrieved:
            raise self.input_error("input must be a retrieve_passages result.", argument="input")
        months = Counter(str(p.get("date", ""))[:7] for p in retrieved["passages"])
        return {
            "months": dict(sorted(months.items())),
            "provenance": self.provenance({}, derived_from=[run_id_of(retrieved)]),
        }
```

Conventions that keep a new tool composable with the rest:

- **Take upstream results as `input`** and put their `run_id_of(...)` into `derived_from`,
  so lineage stays complete.
- **Return a `provenance` block** from `run()` with every setting that shaped the result.
- **Name it `<prefix><object>[_by_<dimension>][_for_<purpose>]`.** `register_tool` rejects
  names without a known prefix.
- **Keep `run()` free of display concerns.** Override `render(data)` if an agent frontend
  should show the result as a table or markdown.

## Comparing a variant with the baseline

Run both over the same test set and put their metrics side by side. Provenance tells the
rows apart, and `anatoolbox.provenance.lineage` turns a list of results into a table of
which tool, with which settings, produced what from what. The course notebook
(`notebooks/course_project_skeleton.ipynb`) does this for every step.
