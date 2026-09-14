"""ingest_corpus -> retrieve_passages, including the recordset chain."""

import pytest

from anatoolbox import ToolContext
from anatoolbox.corpus import clear_corpora, configure_embedder, get_corpus
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.ingest.ingest_corpus import IngestCorpusTool
from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool
from anatoolbox.memory.facade import Memory
from anatoolbox.memory.policy import DefaultMemoryPolicy
from anatoolbox.memory.store import InMemoryRecordsetStore

CSV = (
    "id,title,date,content\n"
    'd1,Agentic web,2024-10-01,"autonomous agents and web standards"\n'
    'd2,AI chips,2025-01-15,"accelerators and data centers"\n'
    'd3,Models,2025-03-20,"foundation model families"\n'
    'd4,Browsing,2025-06-30,"agents browsing the web"\n'
)


def _embedder(texts):
    vocab = ["agents", "web", "accelerators", "model"]
    return [[float(t.lower().count(w)) for w in vocab] for t in texts]


@pytest.fixture(autouse=True)
def clean():
    clear_corpora()
    configure_embedder(_embedder)
    yield
    clear_corpora()
    configure_embedder(None)


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "ai_media.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


@pytest.fixture
def ctx():
    memory = Memory(
        InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="aimedia", session_id="s1"
    )
    return ToolContext(project="aimedia", session_id="s1", recordsets=memory)


class TestIngest:
    def test_registers_a_queryable_corpus(self, corpus_file, ctx):
        out = IngestCorpusTool().run(
            {"path": str(corpus_file), "text_field": "content"}, context=ctx
        )
        assert out["records"] == 4
        assert get_corpus("ai_media") is not None
        assert out["handle"] == "corpus_1"

    def test_stores_ids_by_reference_not_bodies(self, corpus_file, ctx):
        IngestCorpusTool().run({"path": str(corpus_file), "text_field": "content"}, context=ctx)
        record = ctx.recordsets.get("corpus_1")
        assert record.ref["mode"] == "reference"
        assert record.ref["ids"] == ["d1", "d2", "d3", "d4"]
        assert "records" not in record.ref, "document bodies must not enter recordset metadata"

    def test_works_without_recordset_memory(self, corpus_file):
        """Pipelines and notebooks pass no memory; the tool must still run."""
        bare = ToolContext(project="p", session_id="s")
        out = IngestCorpusTool().run(
            {"path": str(corpus_file), "text_field": "content"}, context=bare
        )
        assert out["handle"] is None and out["records"] == 4

    def test_missing_path_is_a_tool_input_error(self, ctx):
        with pytest.raises(ToolInputError):
            IngestCorpusTool().run({}, context=ctx)

    def test_bad_path_is_a_tool_input_error(self, ctx):
        with pytest.raises(ToolInputError):
            IngestCorpusTool().run({"path": "/nope/missing.csv"}, context=ctx)


class TestRetrieve:
    @pytest.fixture(autouse=True)
    def ingested(self, corpus_file, ctx):
        IngestCorpusTool().run({"path": str(corpus_file), "text_field": "content"}, context=ctx)

    def test_binds_the_newest_corpus_when_no_input_given(self, ctx):
        out = RetrievePassagesTool().run({"query": "agents web"}, context=ctx)
        assert out["input_handle"] == "corpus_1"
        assert out["corpus"] == "ai_media"

    def test_records_lineage_back_to_the_corpus(self, ctx):
        out = RetrievePassagesTool().run({"query": "agents web"}, context=ctx)
        assert ctx.recordsets.get(out["handle"]).derived_from == ["corpus_1"]

    @pytest.mark.parametrize("strategy", ["sparse", "dense", "hybrid"])
    def test_all_strategies_run_and_are_labelled(self, ctx, strategy):
        out = RetrievePassagesTool().run(
            {"query": "agents web", "strategy": strategy, "size": 2}, context=ctx
        )
        assert out["strategy"] == strategy
        assert out["returned"] <= 2
        assert [p["rank"] for p in out["passages"]] == list(range(1, out["returned"] + 1))

    def test_strategies_can_disagree(self, ctx):
        """If they always agreed there would be nothing to compare."""
        tool = RetrievePassagesTool()
        sparse = tool.run({"query": "accelerators", "strategy": "sparse", "size": 4}, context=ctx)
        dense = tool.run({"query": "agents", "strategy": "dense", "size": 4}, context=ctx)
        assert [p["id"] for p in sparse["passages"]] != [p["id"] for p in dense["passages"]]

    def test_passages_omit_the_full_text_field(self, ctx):
        out = RetrievePassagesTool().run({"query": "agents"}, context=ctx)
        assert "content" not in out["passages"][0]
        assert out["passages"][0]["snippet"]

    def test_explicit_corpus_name_bypasses_memory(self, ctx):
        out = RetrievePassagesTool().run({"query": "agents", "corpus": "ai_media"}, context=ctx)
        assert out["input_handle"] is None

    def test_unknown_strategy_rejected(self, ctx):
        with pytest.raises(ToolInputError):
            RetrievePassagesTool().run({"query": "x", "strategy": "magic"}, context=ctx)

    def test_missing_query_rejected(self, ctx):
        with pytest.raises(ToolInputError):
            RetrievePassagesTool().run({"strategy": "sparse"}, context=ctx)

    def test_no_corpus_and_no_memory_is_explained(self):
        with pytest.raises(ToolInputError, match="ingest_corpus"):
            RetrievePassagesTool().run(
                {"query": "x"}, context=ToolContext(project="p", session_id="s")
            )

    def test_execute_emits_a_table_render(self, ctx):
        import json

        payload = json.loads(RetrievePassagesTool().execute({"query": "agents web"}, context=ctx))
        assert payload["render"]["render_type"] == "table"
        assert payload["render"]["columns"] == ["rank", "score", "id", "snippet"]


class TestTemporalFiltering:
    @pytest.fixture(autouse=True)
    def ingested(self, corpus_file, ctx):
        IngestCorpusTool().run({"path": str(corpus_file), "text_field": "content"}, context=ctx)

    def test_date_range_is_inclusive(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "agents web", "date_from": "2025-06-30", "date_to": "2025-06-30"},
            context=ctx,
        )
        assert [p["id"] for p in out["passages"]] == ["d4"]

    def test_open_ended_ranges(self, ctx):
        tool = RetrievePassagesTool()
        before = tool.run({"query": "agents web", "date_to": "2024-12-31"}, context=ctx)
        after = tool.run({"query": "agents web", "date_from": "2025-01-01"}, context=ctx)
        assert {p["id"] for p in before["passages"]} == {"d1"}
        assert {p["id"] for p in after["passages"]} == {"d4"}

    @pytest.mark.parametrize("strategy", ["sparse", "dense", "hybrid"])
    def test_every_strategy_respects_the_range(self, ctx, strategy):
        out = RetrievePassagesTool().run(
            {
                "query": "agents web model accelerators",
                "strategy": strategy,
                "date_from": "2025-01-01",
                "size": 10,
            },
            context=ctx,
        )
        assert out["passages"]
        assert all(p["date"] >= "2025-01-01" for p in out["passages"])

    def test_range_is_recorded_in_provenance(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "agents", "date_from": "2025-01-01"}, context=ctx
        )
        assert ctx.recordsets.get(out["handle"]).args["date_from"] == "2025-01-01"
        assert out["date_from"] == "2025-01-01" and out["date_to"] is None

    def test_unconstrained_call_records_no_dates(self, ctx):
        out = RetrievePassagesTool().run({"query": "agents"}, context=ctx)
        args = ctx.recordsets.get(out["handle"]).args
        assert "date_from" not in args and "date_to" not in args

    def test_invalid_date_is_a_tool_input_error(self, ctx):
        with pytest.raises(ToolInputError, match="ISO date"):
            RetrievePassagesTool().run({"query": "agents", "date_from": "last spring"}, context=ctx)

    def test_record_without_a_date_is_excluded_once_a_range_is_set(self, tmp_path, ctx):
        path = tmp_path / "undated.csv"
        path.write_text(
            "id,date,content\nu1,,agents web\nu2,2025-02-01,agents web\n", encoding="utf-8"
        )
        IngestCorpusTool().run({"path": str(path), "text_field": "content"}, context=ctx)
        out = RetrievePassagesTool().run(
            {"query": "agents", "date_from": "2025-01-01"}, context=ctx
        )
        assert [p["id"] for p in out["passages"]] == ["u2"]


class TestRecencyAndFilters:
    @pytest.fixture(autouse=True)
    def ingested(self, corpus_file, ctx):
        IngestCorpusTool().run({"path": str(corpus_file), "text_field": "content"}, context=ctx)

    def test_recency_promotes_newer_passages(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "agents web", "recency_half_life_days": 30}, context=ctx
        )
        assert out["passages"][0]["id"] == "d4"

    def test_ages_are_measured_from_the_newest_date_in_the_corpus(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "agents web", "recency_half_life_days": 30}, context=ctx
        )
        assert out["recency_reference_date"] == "2025-06-30"

    def test_reference_date_can_be_set(self, ctx):
        out = RetrievePassagesTool().run(
            {
                "query": "agents web",
                "recency_half_life_days": 30,
                "recency_reference_date": "2024-10-01",
            },
            context=ctx,
        )
        assert out["recency_reference_date"] == "2024-10-01"

    def test_without_recency_nothing_is_recorded(self, ctx):
        out = RetrievePassagesTool().run({"query": "agents web"}, context=ctx)
        assert out["recency_half_life_days"] is None and out["recency_reference_date"] is None
        assert "recency_half_life_days" not in ctx.recordsets.get(out["handle"]).args

    def test_recency_is_recorded_in_provenance(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "agents web", "recency_half_life_days": 30}, context=ctx
        )
        args = ctx.recordsets.get(out["handle"]).args
        assert args["recency_half_life_days"] == 30
        assert args["recency_reference_date"] == "2025-06-30"

    @pytest.mark.parametrize("value", [0, -5, "soon", True])
    def test_invalid_half_life(self, ctx, value):
        with pytest.raises(ToolInputError, match="recency_half_life_days"):
            RetrievePassagesTool().run(
                {"query": "agents", "recency_half_life_days": value}, context=ctx
            )

    def test_undated_records_sink_once_recency_is_on(self, tmp_path, ctx):
        path = tmp_path / "undated.csv"
        path.write_text(
            "id,date,content\nu1,,agents web agents web\nu2,2025-02-01,agents web\n",
            encoding="utf-8",
        )
        IngestCorpusTool().run({"path": str(path), "text_field": "content"}, context=ctx)
        out = RetrievePassagesTool().run(
            {"query": "agents web", "recency_half_life_days": 30}, context=ctx
        )
        assert [p["id"] for p in out["passages"]] == ["u2", "u1"]

    def test_filters_match_field_values(self, ctx):
        out = RetrievePassagesTool().run(
            {
                "query": "agents web model accelerators",
                "filters": {"title": ["Browsing", "Models"]},
            },
            context=ctx,
        )
        assert out["passages"]
        assert {p["id"] for p in out["passages"]} <= {"d3", "d4"}

    def test_filters_combine_with_a_date_range(self, ctx):
        out = RetrievePassagesTool().run(
            {
                "query": "agents web model accelerators",
                "filters": {"title": ["Browsing", "Models"]},
                "date_to": "2025-04-01",
            },
            context=ctx,
        )
        assert [p["id"] for p in out["passages"]] == ["d3"]

    def test_filters_match_any_value_of_a_list_field(self, tmp_path, ctx):
        path = tmp_path / "tagged.csv"
        path.write_text(
            "id,date,tags,content\n"
            "t1,2025-01-01,\"['GPU', 'Policy']\",agents web\n"
            "t2,2025-01-02,\"['Robotics']\",agents web\n",
            encoding="utf-8",
        )
        IngestCorpusTool().run({"path": str(path), "text_field": "content"}, context=ctx)
        out = RetrievePassagesTool().run(
            {"query": "agents", "filters": {"tags": "GPU"}}, context=ctx
        )
        assert [p["id"] for p in out["passages"]] == ["t1"]

    def test_filters_are_recorded_in_provenance(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "agents", "filters": {"title": "Browsing"}}, context=ctx
        )
        assert ctx.recordsets.get(out["handle"]).args["filters"] == {"title": ["Browsing"]}

    def test_invalid_filters_are_tool_input_errors(self, ctx):
        with pytest.raises(ToolInputError, match="filters"):
            RetrievePassagesTool().run({"query": "agents", "filters": ["title"]}, context=ctx)


class TestMultiQuery:
    @pytest.fixture(autouse=True)
    def ingested(self, corpus_file, ctx):
        IngestCorpusTool().run({"path": str(corpus_file), "text_field": "content"}, context=ctx)

    def test_a_single_query_finds_only_its_topic(self, ctx):
        out = RetrievePassagesTool().run({"query": "accelerators"}, context=ctx)
        assert {p["id"] for p in out["passages"]} == {"d2"}
        assert out["queries"] is None

    def test_extra_queries_are_ranked_and_fused(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "accelerators", "queries": ["foundation model"]}, context=ctx
        )
        assert {"d2", "d3"} <= {p["id"] for p in out["passages"]}
        assert out["queries"] == ["accelerators", "foundation model"]

    def test_repeated_phrasings_are_ignored(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "accelerators", "queries": ["Accelerators"]}, context=ctx
        )
        assert out["queries"] is None

    def test_queries_are_recorded_in_provenance(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "accelerators", "queries": ["foundation model"]}, context=ctx
        )
        assert ctx.recordsets.get(out["handle"]).args["queries"] == [
            "accelerators",
            "foundation model",
        ]

    def test_fusion_respects_size(self, ctx):
        out = RetrievePassagesTool().run(
            {"query": "agents web", "queries": ["foundation model", "accelerators"], "size": 2},
            context=ctx,
        )
        assert out["returned"] == 2

    def test_invalid_queries_are_tool_input_errors(self, ctx):
        with pytest.raises(ToolInputError, match="queries"):
            RetrievePassagesTool().run({"query": "agents", "queries": [1, 2]}, context=ctx)


class TestStoredQueries:
    @pytest.fixture(autouse=True)
    def ingested(self, corpus_file, ctx):
        IngestCorpusTool().run({"path": str(corpus_file), "text_field": "content"}, context=ctx)

    @staticmethod
    def store(ctx, texts):
        from anatoolbox.memory import value_ref

        return ctx.recordsets.remember(
            object_type="queries",
            stage="gather",
            produced_by="rewrite_query_for_retrieval",
            args={"question": "q"},
            ref=value_ref([{"query": text, "purpose": ""} for text in texts]),
            count=len(texts),
            summary="stored queries",
        ).handle

    def test_stored_queries_are_fused_and_recorded_in_lineage(self, ctx):
        handle = self.store(ctx, ["foundation model"])
        out = RetrievePassagesTool().run(
            {"query": "accelerators", "queries_input": handle}, context=ctx
        )
        assert {"d2", "d3"} <= {p["id"] for p in out["passages"]}
        assert out["queries_input_handle"] == handle
        assert ctx.recordsets.get(out["handle"]).derived_from == ["corpus_1", handle]

    def test_an_unknown_queries_handle_is_a_tool_input_error(self, ctx):
        with pytest.raises(ToolInputError):
            RetrievePassagesTool().run(
                {"query": "agents", "queries_input": "queries_9"}, context=ctx
            )

    def test_queries_input_without_memory_is_explained(self, ctx):
        with pytest.raises(ToolInputError, match="queries_input"):
            RetrievePassagesTool().run(
                {"query": "agents", "corpus": "ai_media", "queries_input": "queries_1"},
                context=ToolContext(project="p", session_id="s"),
            )
