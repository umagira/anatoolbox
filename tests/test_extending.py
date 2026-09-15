"""Extending the shipped tools by subclassing: override one hook, inherit the rest."""

import re

import pytest

from anatoolbox import ToolContext
from anatoolbox.corpus import LocalCorpus, clear_corpora, get_corpus, register_corpus
from anatoolbox.enrich.synthesize.synthesize_answer import SynthesizeAnswerTool
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.ingest.ingest_corpus import IngestCorpusTool
from anatoolbox.gather.rerank.rerank_passages import RerankPassagesTool
from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool
from anatoolbox.gather.rewrite.rewrite_query_for_retrieval import RewriteQueryForRetrievalTool
from anatoolbox.preprocess.chunk.chunk_by_size import ChunkBySizeTool
from anatoolbox.registry import ToolRegistrationError, register_tool, resolve_tools
from anatoolbox.tool import with_properties

CTX = ToolContext()


@pytest.fixture(autouse=True)
def articles():
    clear_corpora()
    corpus = register_corpus(
        LocalCorpus.from_records(
            [
                {
                    "id": "a1",
                    "title": "Chips",
                    "date": "2025-01-10",
                    "text": "Nvidia ships a new GPU. It is fast. Data centers want it.",
                },
                {
                    "id": "a2",
                    "title": "Agents",
                    "date": "2025-06-01",
                    "text": "Agents browse the web. Standards for agents emerge.",
                },
                {
                    "id": "a3",
                    "title": "Models",
                    "date": "2024-11-20",
                    "text": "A new foundation model beats the GPU benchmark.",
                },
            ],
            name="articles",
        )
    )
    yield corpus
    clear_corpora()


# --- chunking ----------------------------------------------------------------


class ChunkBySentence(ChunkBySizeTool):
    tool_name = "chunk_by_sentence"
    description = "Split records into chunks of up to max_sentences sentences."
    input_schema = with_properties(
        ChunkBySizeTool.input_schema, {"max_sentences": {"type": "integer", "minimum": 1}}
    )

    def settings(self, args):
        return {
            **super().settings(args),
            "max_sentences": self.int_arg(args, "max_sentences", 2, minimum=1),
        }

    def split(self, text, settings):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        n = settings["max_sentences"]
        return [
            {"text": " ".join(sentences[i : i + n]), "sentences": len(sentences[i : i + n])}
            for i in range(0, len(sentences), n)
        ]


class TestChunkSubclass:
    def test_overriding_split_changes_the_chunks_and_keeps_the_rest(self, articles):
        out = ChunkBySentence().run(
            {"input": {"corpus": "articles"}, "max_sentences": 2}, context=CTX
        )
        chunks = get_corpus(out["corpus"])
        a1 = [c for c in chunks.records if c["source_id"] == "a1"]
        assert [c["chunk_text"] for c in a1] == [
            "Nvidia ships a new GPU. It is fast.",
            "Data centers want it.",
        ]
        # Extra keys from split survive; tokens default to whitespace words; metadata is kept.
        assert a1[0]["sentences"] == 2 and a1[0]["tokens"] == 8
        assert a1[0]["title"] == "Chips" and a1[0]["chunk_count"] == 2

    def test_provenance_and_corpus_name_identify_the_variant(self, articles):
        base = ChunkBySizeTool().run({"corpus": "articles", "size": 4}, context=CTX)
        variant = ChunkBySentence().run({"corpus": "articles", "size": 4}, context=CTX)
        assert base["provenance"]["tool"] == "chunk_by_size"
        assert variant["provenance"]["tool"] == "chunk_by_sentence"
        assert variant["provenance"]["settings"]["max_sentences"] == 2
        assert variant["corpus"] != base["corpus"]
        assert "by_sentence" in variant["corpus"]
        get_corpus(base["corpus"])  # the baseline corpus was not replaced

    def test_new_argument_is_checked_and_in_the_schema(self):
        tool = ChunkBySentence()
        assert tool.schema.name == "chunk_by_sentence"
        assert "max_sentences" in tool.schema.input_schema["properties"]
        assert "max_sentences" not in ChunkBySizeTool().schema.input_schema["properties"]
        with pytest.raises(ToolInputError, match="max_sentences") as error:
            tool.run({"corpus": "articles", "max_sentences": 0}, context=CTX)
        assert error.value.tool_name == "chunk_by_sentence"

    def test_overriding_contextualize(self, articles):
        class ChunkWithDate(ChunkBySizeTool):
            tool_name = "chunk_with_date"

            def contextualize(self, record, chunk_text, settings):
                return f"{record['title']} ({record['date']})"

        out = ChunkWithDate().run({"corpus": "articles", "size": 50}, context=CTX)
        first = get_corpus(out["corpus"]).records[0]
        assert first["context"] == "Chips (2025-01-10)"
        assert first["text"].startswith("Chips (2025-01-10)\n\n")


# --- registry ------------------------------------------------------------------


class TestRegistering:
    def test_a_renamed_subclass_registers_next_to_the_baseline(self):
        register_tool(ChunkBySizeTool())
        register_tool(ChunkBySentence())
        names = [t.schema.name for t in resolve_tools(["chunk_by_size", "chunk_by_sentence"])]
        assert names == ["chunk_by_size", "chunk_by_sentence"]

    def test_a_subclass_that_forgets_to_rename_is_refused(self):
        class Unnamed(ChunkBySizeTool):
            def split(self, text, settings):
                return [{"text": text}]

        register_tool(ChunkBySizeTool())
        with pytest.raises(ToolRegistrationError, match="already registered"):
            register_tool(Unnamed())

    def test_the_name_must_keep_the_prefix(self):
        class Misnamed(ChunkBySizeTool):
            tool_name = "sentence_chunker"

        with pytest.raises(ToolRegistrationError, match="prefix"):
            register_tool(Misnamed())


# --- retrieval -----------------------------------------------------------------


class TestRetrieveSubclass:
    def test_overriding_rank(self, articles):
        class RetrieveNewestFirst(RetrievePassagesTool):
            tool_name = "retrieve_newest_first"

            def rank(self, query, corpus, settings, limit):
                hits = super().rank(query, corpus, settings, limit)
                return sorted(hits, key=lambda h: corpus.records[h.index]["date"], reverse=True)

        out = RetrieveNewestFirst().run({"query": "GPU", "corpus": "articles"}, context=CTX)
        assert [p["id"] for p in out["passages"]] == ["a1", "a3"]
        assert out["provenance"]["tool"] == "retrieve_newest_first"

    def test_overriding_keep_still_walks_the_ranking(self, articles):
        class RetrieveSince2025(RetrievePassagesTool):
            tool_name = "retrieve_since_2025"

            def keep(self, record, settings):
                return record["date"] >= "2025-01-01" and super().keep(record, settings)

        out = RetrieveSince2025().run(
            {"query": "GPU new", "corpus": "articles", "size": 1}, context=CTX
        )
        assert [p["id"] for p in out["passages"]] == ["a1"]

    def test_overriding_boost_applies_without_recency_arguments(self, articles):
        class RetrievePreferAgents(RetrievePassagesTool):
            tool_name = "retrieve_prefer_agents"

            def boost(self, record, settings):
                return 100.0 if record["title"] == "Agents" else 1.0

        out = RetrievePreferAgents().run(
            {"query": "new agents GPU", "corpus": "articles"}, context=CTX
        )
        assert out["passages"][0]["id"] == "a2"

    def test_overriding_fuse(self, articles):
        class RetrieveFirstQueryWins(RetrievePassagesTool):
            tool_name = "retrieve_first_query_wins"

            def fuse(self, rankings, settings):
                return rankings[0]

        out = RetrieveFirstQueryWins().run(
            {"query": "agents", "queries": ["GPU"], "corpus": "articles"}, context=CTX
        )
        assert [p["id"] for p in out["passages"]] == ["a2"]


# --- reranking -----------------------------------------------------------------


class TestRerankSubclass:
    def test_overriding_score_needs_no_model(self, articles):
        class RerankShortestFirst(RerankPassagesTool):
            tool_name = "rerank_shortest_first"

            def score(self, query, texts, settings):
                return [-len(text) for text in texts]

            def reranker(self, settings):
                return "length"

        retrieved = RetrievePassagesTool().run(
            {"query": "GPU agents new", "corpus": "articles"}, context=CTX
        )
        out = RerankShortestFirst().run({"input": retrieved, "keep": 3}, context=CTX)
        lengths = [len(get_corpus("articles").get([p["id"]])[0]["text"]) for p in out["passages"]]
        assert lengths == sorted(lengths)
        assert out["reranker"] == "length"
        assert out["provenance"]["tool"] == "rerank_shortest_first"
        assert out["provenance"]["derived_from"] == [retrieved["provenance"]["run_id"]]

    def test_a_score_of_the_wrong_length_fails_loudly(self, articles):
        class Broken(RerankPassagesTool):
            tool_name = "rerank_broken"

            def score(self, query, texts, settings):
                return [1.0]

        passages = [{"id": "a1", "text": "x"}, {"id": "a2", "text": "y"}]
        with pytest.raises(ValueError, match="1 scores for 2 texts"):
            Broken().run({"query": "q", "passages": passages}, context=CTX)


# --- LLM tools -----------------------------------------------------------------


class TestRewriteSubclass:
    def test_overriding_rewrite_keeps_cleaning_and_provenance(self, fake_llm):
        class RewriteByTemplate(RewriteQueryForRetrievalTool):
            tool_name = "rewrite_query_by_template"

            def rewrite(self, settings, context):
                q = settings["question"]
                return {
                    "queries": [q, f"{q} news", f"{q} news"],
                    "exact_terms": [],
                    "time_range": None,
                }

        out = RewriteByTemplate().run({"question": "AI chips", "max_queries": 3}, context=CTX)
        assert out["query_texts"] == ["AI chips", "AI chips news"]  # duplicates dropped by clean
        assert out["provenance"]["tool"] == "rewrite_query_by_template"
        assert fake_llm.calls == []


class TestSynthesizeSubclass:
    def test_overriding_generate_and_curate(self, fake_llm):
        class SynthesizeFirstSourceOnly(SynthesizeAnswerTool):
            tool_name = "synthesize_answer_from_first_source"

            def curate(self, passages, settings):
                sources, duplicates, over = super().curate(passages[:1], settings)
                return sources, duplicates, over

            def generate(self, system, user, settings, context):
                assert "[S1]" in user and "[S2]" not in user
                return "GPUs are shipping [S1]. Also [S7]."

        passages = [{"id": "a1", "text": "GPU news"}, {"id": "a2", "text": "Agent news"}]
        out = SynthesizeFirstSourceOnly().run(
            {"question": "What ships?", "passages": passages}, context=CTX
        )
        assert out["cited"] == ["S1"] and out["unknown_citations"] == ["S7"]
        assert out["provenance"]["tool"] == "synthesize_answer_from_first_source"
        assert fake_llm.calls == []

    def test_overriding_the_prompts(self, fake_llm):
        class SynthesizeStepByStep(SynthesizeAnswerTool):
            tool_name = "synthesize_answer_step_by_step"

            def system_prompt(self, settings):
                return super().system_prompt(settings) + "\n- Think step by step first."

        fake_llm.replies.append("It ships [S1].")
        SynthesizeStepByStep().run(
            {"question": "What ships?", "passages": [{"id": "a1", "text": "GPU news"}]}, context=CTX
        )
        assert fake_llm.system_prompt().endswith("Think step by step first.")


# --- ingest --------------------------------------------------------------------


class TestIngestSubclass:
    def test_overriding_prepare(self, tmp_path):
        path = tmp_path / "articles.jsonl"
        path.write_text(
            '{"id": "1", "text": "short"}\n{"id": "2", "text": "a much longer article body"}\n',
            encoding="utf-8",
        )

        class IngestLongArticles(IngestCorpusTool):
            tool_name = "ingest_long_articles"

            def prepare(self, corpus, settings):
                corpus.records = [r for r in corpus.records if len(r["text"]) > 10]
                corpus.refresh()
                return corpus

        out = IngestLongArticles().run({"path": str(path)}, context=CTX)
        assert out["records"] == 1 and get_corpus("articles").ids == ["2"]
        assert out["provenance"]["tool"] == "ingest_long_articles"
        assert out["provenance"]["settings"]["source_file"] == "articles.jsonl"
        assert "path" not in out["provenance"]["settings"]
