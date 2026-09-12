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
    "id,title,content\n"
    'd1,Agentic web,"autonomous agents and web standards"\n'
    'd2,AI chips,"accelerators and data centers"\n'
    'd3,Models,"foundation model families"\n'
    'd4,Browsing,"agents browsing the web"\n'
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
