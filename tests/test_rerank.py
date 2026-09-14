"""rerank_passages and the reranker seam."""

import pytest

from anatoolbox import ToolContext
from anatoolbox.corpus import LocalCorpus, clear_corpora, local_ref, register_corpus
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.rerank.rerank_passages import SNIPPET_CHARS, RerankPassagesTool
from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool
from anatoolbox.memory.facade import Memory
from anatoolbox.memory.policy import DefaultMemoryPolicy
from anatoolbox.memory.store import InMemoryRecordsetStore
from anatoolbox.reranking import (
    DEFAULT_RERANK_MODEL,
    configure_reranker,
    rerank_texts,
    reranker_label,
)

PADDING = "filler " * 120
DOCS = [
    {"id": "d1", "date": "2025-01-01", "text": "agents web agents web agents web"},
    {
        "id": "d2",
        "date": "2025-02-01",
        "text": "agents web. " + PADDING + "The decisive detail: protocol.",
    },
    {"id": "d3", "date": "2025-03-01", "text": "agents on the web today"},
]


def keyword_reranker(query, texts):
    """Scores by occurrences of the query's last word — enough to test ordering."""
    term = query.split()[-1].lower()
    return [float(text.lower().count(term)) for text in texts]


@pytest.fixture(autouse=True)
def reranker():
    clear_corpora()
    configure_reranker(keyword_reranker, label="keyword-test")
    yield
    configure_reranker(None)
    clear_corpora()


@pytest.fixture
def ctx():
    memory = Memory(InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="p", session_id="s")
    corpus = register_corpus(LocalCorpus.from_records(DOCS, name="docs"))
    memory.remember(
        object_type="corpus",
        stage="gather",
        produced_by="ingest_corpus",
        args={"corpus": "docs"},
        ref=local_ref(corpus="docs", ids=corpus.ids),
        count=len(corpus),
        summary="3 docs",
    )
    return ToolContext(project="p", session_id="s", recordsets=memory)


def retrieve(ctx, query="agents web protocol", size=3):
    return RetrievePassagesTool().run({"query": query, "size": size}, context=ctx)


def test_reranks_on_the_full_text_not_the_snippet(ctx):
    retrieve(ctx)
    out = RerankPassagesTool().run({"keep": 3}, context=ctx)
    top = out["passages"][0]
    assert top["id"] == "d2"
    assert "protocol" not in top["snippet"].lower(), "the deciding term sits past the snippet"
    assert DOCS[1]["text"].lower().index("protocol") > SNIPPET_CHARS


def test_reuses_the_query_the_candidates_were_retrieved_with(ctx):
    retrieve(ctx)
    assert RerankPassagesTool().run({}, context=ctx)["query"] == "agents web protocol"


def test_an_explicit_query_wins(ctx):
    retrieve(ctx)
    out = RerankPassagesTool().run({"query": "agents today", "keep": 1}, context=ctx)
    assert out["passages"][0]["id"] == "d3"


def test_keep_truncates_and_rank_movement_is_reported(ctx):
    retrieve(ctx)
    out = RerankPassagesTool().run({"keep": 1}, context=ctx)
    top = out["passages"][0]
    assert (out["candidates"], out["returned"], top["rank"]) == (3, 1, 1)
    assert top["rank_change"] == top["retrieval_rank"] - 1


def test_lineage_and_provenance(ctx):
    candidates = retrieve(ctx)
    out = RerankPassagesTool().run({"keep": 2}, context=ctx)
    record = ctx.recordsets.get(out["handle"])
    assert record.object_type == "passages"
    assert record.derived_from == [candidates["handle"]]
    assert record.args["reranker"] == "keyword-test" and record.args["keep"] == 2
    assert record.ref["ids"] == [p["id"] for p in out["passages"]]
    assert record.ref["mode"] == "reference"


def test_binds_an_explicit_candidate_handle(ctx):
    first = retrieve(ctx)
    retrieve(ctx, query="today")
    out = RerankPassagesTool().run({"input": first["handle"], "keep": 1}, context=ctx)
    assert out["input_handle"] == first["handle"]
    assert out["query"] == "agents web protocol"


def test_max_chars_bounds_what_the_reranker_reads(ctx):
    retrieve(ctx)
    out = RerankPassagesTool().run({"keep": 3, "max_chars": 100}, context=ctx)
    assert [p["rank_change"] for p in out["passages"]] == [0, 0, 0], "no signal left, order kept"


def test_explicit_passages_need_no_memory():
    out = RerankPassagesTool().run(
        {
            "query": "find protocol",
            "passages": [{"id": "x", "text": "no"}, {"id": "y", "text": "protocol here"}],
        },
        context=ToolContext(project="p", session_id="s"),
    )
    assert [p["id"] for p in out["passages"]] == ["y", "x"]
    assert out["handle"] is None


def test_explicit_passages_without_a_query_are_explained():
    with pytest.raises(ToolInputError, match="No query"):
        RerankPassagesTool().run(
            {"passages": [{"id": "x", "text": "t"}]},
            context=ToolContext(project="p", session_id="s"),
        )


def test_no_candidates_and_no_memory_is_explained():
    with pytest.raises(ToolInputError, match="retrieve_passages"):
        RerankPassagesTool().run({"query": "q"}, context=ToolContext(project="p", session_id="s"))


def test_an_empty_retrieval_leaves_nothing_to_rerank(ctx):
    """Memory does not store an empty recordset, so there is nothing to bind."""
    assert retrieve(ctx, query="zzzz")["returned"] == 0
    with pytest.raises(ToolInputError, match="No 'passages' data is stored"):
        RerankPassagesTool().run({}, context=ctx)


def test_an_empty_candidate_list_is_explained():
    with pytest.raises(ToolInputError, match="no candidate"):
        RerankPassagesTool().run(
            {"query": "q", "passages": []}, context=ToolContext(project="p", session_id="s")
        )


@pytest.mark.parametrize("bad", [{"keep": 0}, {"keep": "five"}, {"max_chars": -1}])
def test_invalid_numbers_are_tool_input_errors(ctx, bad):
    retrieve(ctx)
    with pytest.raises(ToolInputError):
        RerankPassagesTool().run(bad, context=ctx)


def test_malformed_explicit_passages_are_rejected():
    with pytest.raises(ToolInputError, match="'id' and 'text'"):
        RerankPassagesTool().run(
            {"query": "q", "passages": [{"id": "x"}]},
            context=ToolContext(project="p", session_id="s"),
        )


def test_a_reranker_must_return_one_score_per_text():
    with pytest.raises(ValueError, match="one score per text"):
        rerank_texts("q", ["a", "b"], reranker=lambda q, t: [1.0])


def test_reranker_label_falls_back_to_the_function_name():
    configure_reranker(keyword_reranker)
    assert reranker_label() == "keyword_reranker"
    configure_reranker(None)
    assert reranker_label() == DEFAULT_RERANK_MODEL


def test_max_per_source_diversifies_the_kept_passages():
    """After chunking, several chunks of one article can otherwise fill every slot."""
    passages = [
        {"id": "x#0", "source_id": "x", "text": "protocol protocol protocol"},
        {"id": "x#1", "source_id": "x", "text": "protocol protocol"},
        {"id": "y#0", "source_id": "y", "text": "protocol"},
    ]
    out = RerankPassagesTool().run(
        {"query": "find protocol", "passages": passages, "keep": 2, "max_per_source": 1},
        context=ToolContext(project="p", session_id="s"),
    )
    assert [p["id"] for p in out["passages"]] == ["x#0", "y#0"]
    assert out["skipped_over_source_limit"] == 1


def test_without_a_source_limit_the_best_scores_win(ctx):
    retrieve(ctx)
    out = RerankPassagesTool().run({"keep": 3}, context=ctx)
    assert out["skipped_over_source_limit"] == 0


def test_invalid_max_per_source(ctx):
    retrieve(ctx)
    with pytest.raises(ToolInputError, match="max_per_source"):
        RerankPassagesTool().run({"max_per_source": 0}, context=ctx)
