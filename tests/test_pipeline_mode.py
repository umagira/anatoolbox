"""Pipeline mode: tools chained by passing results, no memory, portable provenance."""

import csv
import json

import pytest

from anatoolbox import ToolContext
from anatoolbox.corpus import LocalCorpus, clear_corpora, register_corpus
from anatoolbox.enrich.synthesize.synthesize_answer import SynthesizeAnswerTool
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.ingest.ingest_corpus import IngestCorpusTool
from anatoolbox.gather.rerank.rerank_passages import RerankPassagesTool
from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool
from anatoolbox.gather.rewrite.rewrite_query_for_retrieval import RewriteQueryForRetrievalTool
from anatoolbox.memory.facade import Memory
from anatoolbox.memory.policy import DefaultMemoryPolicy
from anatoolbox.memory.store import InMemoryRecordsetStore
from anatoolbox.preprocess.chunk.chunk_by_size import ChunkBySizeTool
from anatoolbox.reranking import configure_reranker

QUESTION = "How do browser agents use the protocol"
PROVENANCE_KEYS = {"run_id", "tool", "anatoolbox_version", "created_at", "settings", "derived_from"}


def keyword_reranker(query, texts):
    term = query.split()[-1].lower()
    return [float(text.lower().count(term)) for text in texts]


@pytest.fixture(autouse=True)
def clean():
    clear_corpora()
    configure_reranker(keyword_reranker, label="keyword-test")
    yield
    configure_reranker(None)
    clear_corpora()


@pytest.fixture
def news_csv(tmp_path):
    path = tmp_path / "news.csv"
    rows = [
        (
            "n1",
            "Agents and risk",
            "2025-03-01",
            ["Browser agents act on websites.", "Security teams worry about browser agents."],
        ),
        ("n2", "Chips", "2025-02-01", ["Accelerators are scarce."]),
        ("n3", "Protocols", "2025-04-01", ["Browser agents speak a protocol to reach tools."]),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "title", "date", "content"])
        for row_id, title, date, paragraphs in rows:
            writer.writerow([row_id, title, date, str(paragraphs)])
    return path


@pytest.fixture
def long_docs():
    padding = "filler " * 120
    return register_corpus(
        LocalCorpus.from_records(
            [
                {"id": "d1", "title": "Short", "date": "2025-01-01", "text": "browser agents"},
                {
                    "id": "d2",
                    "title": "Long",
                    "date": "2025-02-01",
                    "text": "browser agents. " + padding + "The protocol.",
                },
            ],
            name="docs",
        )
    )


def rewrite_reply(*queries):
    return json.dumps(
        {
            "queries": [{"query": q, "purpose": ""} for q in queries],
            "exact_terms": [],
            "time_range": {"from": None, "to": None},
        }
    )


def test_tool_context_needs_no_arguments():
    ctx = ToolContext()
    assert (ctx.project, ctx.session_id, ctx.recordsets) == ("default", "", None)


def test_the_whole_loop_chains_by_passing_results(news_csv, fake_llm):
    ctx = ToolContext()
    fake_llm.replies = [rewrite_reply("browser agents protocol"), ["Agents speak a protocol [S1]."]]

    articles = IngestCorpusTool().run({"path": str(news_csv), "text_field": "content"}, context=ctx)
    chunks = ChunkBySizeTool().run({"input": articles, "size": 5}, context=ctx)
    rewrites = RewriteQueryForRetrievalTool().run({"question": QUESTION}, context=ctx)
    retrieved = RetrievePassagesTool().run(
        {"query": QUESTION, "input": chunks, "queries_input": rewrites, "size": 5}, context=ctx
    )
    reranked = RerankPassagesTool().run({"input": retrieved, "keep": 3}, context=ctx)
    answer = SynthesizeAnswerTool().run({"question": QUESTION, "input": reranked}, context=ctx)

    run_id = lambda result: result["provenance"]["run_id"]  # noqa: E731
    assert chunks["provenance"]["derived_from"] == [run_id(articles)]
    assert rewrites["provenance"]["derived_from"] == []
    assert retrieved["provenance"]["derived_from"] == [run_id(chunks), run_id(rewrites)]
    assert reranked["provenance"]["derived_from"] == [run_id(retrieved)]
    assert answer["provenance"]["derived_from"] == [run_id(reranked)]

    assert retrieved["queries"] == [QUESTION, "browser agents protocol"]
    assert reranked["query"] == QUESTION, "the query travels with the retrieval result"
    assert "protocol" in reranked["passages"][0]["snippet"].lower()
    assert answer["cited"] == ["S1"]
    for result in (articles, chunks, rewrites, retrieved, reranked, answer):
        assert result["handle"] is None, "no memory, no handles"


def test_passages_from_a_result_are_reranked_on_their_full_text(long_docs):
    ctx = ToolContext()
    retrieved = RetrievePassagesTool().run(
        {"query": "browser agents protocol", "corpus": "docs", "size": 2}, context=ctx
    )
    assert all("text" not in p for p in retrieved["passages"]), "results carry snippets only"
    reranked = RerankPassagesTool().run({"input": retrieved, "keep": 2}, context=ctx)
    assert reranked["passages"][0]["id"] == "d2"
    assert "protocol" not in reranked["passages"][0]["snippet"].lower(), (
        "decided by text past the snippet"
    )


def test_the_answer_prompt_gets_full_text_from_a_result(long_docs, fake_llm):
    ctx = ToolContext()
    fake_llm.replies = [["ok"]]
    retrieved = RetrievePassagesTool().run(
        {"query": "browser agents", "corpus": "docs", "size": 2}, context=ctx
    )
    SynthesizeAnswerTool().run(
        {"question": "q", "input": retrieved, "max_chars_per_passage": 5000}, context=ctx
    )
    assert "The protocol." in fake_llm.user_prompt()


def test_ids_alone_are_enough_when_the_corpus_is_named(long_docs):
    out = RerankPassagesTool().run(
        {"query": "find the protocol", "passages": [{"id": "d1"}, {"id": "d2"}], "corpus": "docs"},
        context=ToolContext(),
    )
    assert out["passages"][0]["id"] == "d2"


def test_ids_without_text_or_corpus_are_explained():
    with pytest.raises(ToolInputError, match="corpus"):
        RerankPassagesTool().run({"query": "q", "passages": [{"id": "x"}]}, context=ToolContext())


def test_unknown_ids_are_named(long_docs):
    with pytest.raises(ToolInputError, match="nope"):
        RerankPassagesTool().run(
            {"query": "q", "passages": [{"id": "nope"}], "corpus": "docs"}, context=ToolContext()
        )


def test_an_input_that_is_not_a_usable_result_is_explained(long_docs):
    with pytest.raises(ToolInputError, match="names its corpus"):
        ChunkBySizeTool().run({"input": {"something": 1}}, context=ToolContext())
    with pytest.raises(ToolInputError, match="result with 'passages'"):
        RerankPassagesTool().run({"query": "q", "input": {"something": 1}}, context=ToolContext())
    with pytest.raises(ToolInputError, match="rewrite_query_for_retrieval"):
        RetrievePassagesTool().run(
            {"query": "q", "corpus": "docs", "queries_input": {"query_texts": []}},
            context=ToolContext(),
        )


def test_an_explicit_corpus_name_wins_over_an_input_result(long_docs, news_csv):
    ctx = ToolContext()
    articles = IngestCorpusTool().run({"path": str(news_csv), "text_field": "content"}, context=ctx)
    out = RetrievePassagesTool().run(
        {"query": "browser agents", "corpus": "docs", "input": articles}, context=ctx
    )
    assert out["corpus"] == "docs"
    assert out["provenance"]["derived_from"] == []


def test_without_memory_nothing_is_bound_implicitly(news_csv):
    ctx = ToolContext()
    IngestCorpusTool().run({"path": str(news_csv), "text_field": "content"}, context=ctx)
    with pytest.raises(ToolInputError, match="No corpus given"):
        RetrievePassagesTool().run({"query": "browser agents"}, context=ctx)


class TestProvenance:
    def test_every_result_carries_portable_provenance(self, news_csv, fake_llm):
        ctx = ToolContext()
        fake_llm.replies = [rewrite_reply("q1"), ["ok [S1]"]]
        articles = IngestCorpusTool().run(
            {"path": str(news_csv), "text_field": "content"}, context=ctx
        )
        chunks = ChunkBySizeTool().run({"input": articles, "size": 5}, context=ctx)
        rewrites = RewriteQueryForRetrievalTool().run({"question": "browser agents"}, context=ctx)
        retrieved = RetrievePassagesTool().run(
            {"query": "browser agents", "input": chunks}, context=ctx
        )
        reranked = RerankPassagesTool().run({"input": retrieved}, context=ctx)
        answer = SynthesizeAnswerTool().run({"question": "q", "input": reranked}, context=ctx)
        for result, tool in [
            (articles, "ingest_corpus"),
            (chunks, "chunk_by_size"),
            (rewrites, "rewrite_query_for_retrieval"),
            (retrieved, "retrieve_passages"),
            (reranked, "rerank_passages"),
            (answer, "synthesize_answer"),
        ]:
            provenance = result["provenance"]
            assert set(provenance) == PROVENANCE_KEYS
            assert provenance["tool"] == tool and provenance["run_id"].startswith(tool + "-")
            json.dumps(provenance)

    def test_run_ids_are_unique(self, long_docs):
        ctx = ToolContext()
        first = RetrievePassagesTool().run({"query": "browser", "corpus": "docs"}, context=ctx)
        second = RetrievePassagesTool().run({"query": "browser", "corpus": "docs"}, context=ctx)
        assert first["provenance"]["run_id"] != second["provenance"]["run_id"]

    def test_settings_include_the_defaults_that_were_applied(self, long_docs, fake_llm):
        ctx = ToolContext()
        fake_llm.replies = [["ok"]]
        retrieved = RetrievePassagesTool().run({"query": "browser", "corpus": "docs"}, context=ctx)
        assert (
            retrieved["provenance"]["settings"]["strategy"],
            retrieved["provenance"]["settings"]["size"],
        ) == ("sparse", 10)
        answer = SynthesizeAnswerTool().run({"question": "q", "input": retrieved}, context=ctx)
        settings = answer["provenance"]["settings"]
        assert (settings["model"], settings["mode"], settings["order"]) == (
            "strong-model",
            "grounded",
            "edges",
        )

    def test_ingest_records_the_file_name_not_its_path(self, news_csv):
        result = IngestCorpusTool().run(
            {"path": str(news_csv), "text_field": "content"}, context=ToolContext()
        )
        assert result["provenance"]["settings"]["source_file"] == "news.csv"
        assert str(news_csv.parent) not in json.dumps(result["provenance"])
        assert result["corpus"] == "news"

    def test_with_memory_provenance_names_the_handles(self, news_csv):
        memory = Memory(
            InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="p", session_id="s"
        )
        ctx = ToolContext(recordsets=memory)
        articles = IngestCorpusTool().run(
            {"path": str(news_csv), "text_field": "content"}, context=ctx
        )
        retrieved = RetrievePassagesTool().run(
            {"query": "browser agents", "input": articles["handle"]}, context=ctx
        )
        reranked = RerankPassagesTool().run({}, context=ctx)
        assert retrieved["handle"] == "passages_1" and retrieved["provenance"]["derived_from"] == [
            "corpus_1"
        ]
        assert reranked["provenance"]["derived_from"] == ["passages_1"]
        assert memory.get(reranked["handle"]).derived_from == ["passages_1"]

    def test_results_passed_with_memory_attached_do_not_enter_recordset_lineage(self, news_csv):
        memory = Memory(
            InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="p", session_id="s"
        )
        ctx = ToolContext(recordsets=memory)
        articles = IngestCorpusTool().run(
            {"path": str(news_csv), "text_field": "content"}, context=ctx
        )
        retrieved = RetrievePassagesTool().run(
            {"query": "browser agents", "input": articles}, context=ctx
        )
        assert memory.get(retrieved["handle"]).derived_from == [], (
            "run ids are not recordset handles"
        )
        assert retrieved["provenance"]["derived_from"] == [articles["provenance"]["run_id"]]
