"""chunk_by_size: the fixed-size baseline."""

import pytest

from anatoolbox import ToolContext
from anatoolbox.corpus import LocalCorpus, clear_corpora, get_corpus, local_ref, register_corpus
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool
from anatoolbox.memory.facade import Memory
from anatoolbox.memory.policy import DefaultMemoryPolicy
from anatoolbox.memory.store import InMemoryRecordsetStore
from anatoolbox.preprocess.chunk import chunk_by_size as module
from anatoolbox.preprocess.chunk.chunk_by_size import (
    ChunkBySizeTool,
    chunk_text_by_size,
    register_contextualizer,
    whitespace_spans,
)


def words(n):
    return " ".join(f"w{i}" for i in range(n))


class TestChunkTextBySize:
    def test_windows_have_the_requested_size_and_cover_the_text(self):
        chunks = chunk_text_by_size(words(10), size=4)
        assert [c["tokens"] for c in chunks] == [4, 4, 2]
        assert [(c["token_start"], c["token_end"]) for c in chunks] == [(0, 4), (4, 8), (8, 10)]
        assert " ".join(c["text"] for c in chunks) == words(10)

    def test_overlap_shares_tokens_between_neighbours(self):
        chunks = chunk_text_by_size(words(10), size=4, overlap=2)
        assert [(c["token_start"], c["token_end"]) for c in chunks] == [
            (0, 4),
            (2, 6),
            (4, 8),
            (6, 10),
        ]

    def test_no_window_is_made_only_of_overlap(self):
        chunks = chunk_text_by_size(words(6), size=4, overlap=2)
        assert [(c["token_start"], c["token_end"]) for c in chunks] == [(0, 4), (2, 6)]

    def test_original_spacing_inside_a_window_is_kept(self):
        text = "one two\n\nthree   four five"
        chunks = chunk_text_by_size(text, size=4)
        assert chunks[0]["text"] == "one two\n\nthree   four"
        assert text[chunks[0]["char_start"] : chunks[0]["char_end"]] == chunks[0]["text"]

    def test_it_ignores_structure_on_purpose(self):
        chunks = chunk_text_by_size("alpha beta.\n\ngamma delta.", size=3)
        assert chunks[0]["text"] == "alpha beta.\n\ngamma", "a baseline cuts through paragraphs"

    def test_a_custom_tokenizer_decides_what_a_token_is(self):
        characters = lambda text: [(i, i + 1) for i, ch in enumerate(text) if not ch.isspace()]  # noqa: E731
        assert [c["text"] for c in chunk_text_by_size("abc de", size=2, tokenizer=characters)] == [
            "ab",
            "c d",
            "e",
        ]

    def test_empty_text_gives_no_chunks(self):
        assert chunk_text_by_size("") == [] and chunk_text_by_size("   \n ") == []

    @pytest.mark.parametrize("size, overlap", [(0, 0), (4, 4), (4, 5), (4, -1)])
    def test_invalid_size_or_overlap(self, size, overlap):
        with pytest.raises(ValueError):
            chunk_text_by_size("a b c", size=size, overlap=overlap)

    def test_whitespace_spans(self):
        assert whitespace_spans(" ab  c\nd ") == [(1, 3), (5, 6), (7, 8)]


ARTICLES = [
    {
        "id": "a1",
        "title": "Agents",
        "date": "2025-05-19",
        "url": "https://e.org/a1",
        "content": ["Agents browse web services.", "MCP servers expose tools to agents."],
    },
    {
        "id": "a2",
        "title": "Chips",
        "date": "2024-11-02",
        "url": "https://e.org/a2",
        "content": ["Accelerators remain scarce."],
    },
    {"id": "a3", "title": "Empty", "date": "2025-01-01", "url": "https://e.org/a3", "content": []},
]


@pytest.fixture(autouse=True)
def clean():
    clear_corpora()
    saved = dict(module._CONTEXTUALIZERS)
    register_corpus(LocalCorpus.from_records(ARTICLES, name="articles", text_field="content"))
    yield
    module._CONTEXTUALIZERS.clear()
    module._CONTEXTUALIZERS.update(saved)
    clear_corpora()


def ingest_result():
    """What ingest_corpus returns, reduced to the fields a consumer uses."""
    return {"corpus": "articles", "provenance": {"run_id": "ingest_corpus-1234abcd"}}


class TestChunkBySizeTool:
    def test_chunks_carry_ids_positions_and_metadata(self):
        out = ChunkBySizeTool().run({"input": ingest_result(), "size": 4}, context=ToolContext())
        assert (out["corpus"], out["articles"], out["empty_articles"], out["chunks"]) == (
            "articles_size4_overlap0",
            2,
            1,
            4,
        )
        chunk = get_corpus(out["corpus"]).get(["a1#1"])[0]
        assert (chunk["source_id"], chunk["chunk_index"], chunk["chunk_count"]) == ("a1", 1, 3)
        assert (chunk["title"], chunk["date"], chunk["url"]) == (
            "Agents",
            "2025-05-19",
            "https://e.org/a1",
        )
        assert (chunk["token_start"], chunk["token_end"], chunk["tokens"]) == (4, 8, 4)
        assert "content" not in chunk

    def test_provenance_records_the_baseline_settings(self):
        out = ChunkBySizeTool().run(
            {"input": ingest_result(), "size": 4, "overlap": 1}, context=ToolContext()
        )
        assert out["provenance"]["derived_from"] == ["ingest_corpus-1234abcd"]
        settings = out["provenance"]["settings"]
        assert (settings["size"], settings["overlap"], settings["tokenizer"]) == (
            4,
            1,
            "whitespace",
        )
        assert settings["contextualize"] == "none"

    def test_retrieval_searches_the_chunks(self):
        chunks = ChunkBySizeTool().run({"corpus": "articles", "size": 4}, context=ToolContext())
        out = RetrievePassagesTool().run(
            {"query": "MCP servers", "input": chunks}, context=ToolContext()
        )
        assert out["passages"][0]["source_id"] == "a1"
        assert out["provenance"]["derived_from"] == [chunks["provenance"]["run_id"]]

    def test_with_memory_it_records_a_corpus_recordset(self):
        memory = Memory(
            InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="p", session_id="s"
        )
        memory.remember(
            object_type="corpus",
            stage="gather",
            produced_by="ingest_corpus",
            args={"corpus": "articles"},
            ref=local_ref(corpus="articles", ids=["a1", "a2", "a3"]),
            count=3,
            summary="articles",
        )
        out = ChunkBySizeTool().run({"size": 4}, context=ToolContext(recordsets=memory))
        assert out["handle"] == "corpus_2" and memory.get("corpus_2").derived_from == ["corpus_1"]
        assert out["provenance"]["derived_from"] == ["corpus_1"]

    @pytest.mark.parametrize(
        "bad, match",
        [
            ({"size": 0}, "size"),
            ({"size": 4, "overlap": 4}, "overlap must be smaller"),
            ({"size": 4, "overlap": -1}, "overlap"),
            ({"size": "big"}, "size"),
            ({"size": 4, "name": "articles"}, "different name"),
        ],
    )
    def test_invalid_arguments(self, bad, match):
        with pytest.raises(ToolInputError, match=match):
            ChunkBySizeTool().run({"corpus": "articles", **bad}, context=ToolContext())

    def test_a_missing_tokenizer_dependency_is_explained(self, monkeypatch):
        def unavailable(name):
            raise ImportError("no transformers")

        monkeypatch.setattr(module, "_load_hf_tokenizer", unavailable)
        with pytest.raises(ToolInputError, match="embeddings"):
            ChunkBySizeTool().run(
                {"corpus": "articles", "tokenizer": "some/model"}, context=ToolContext()
            )

    def test_an_unloadable_tokenizer_is_explained(self, monkeypatch):
        def broken(name):
            raise OSError("not found")

        monkeypatch.setattr(module, "_load_hf_tokenizer", broken)
        with pytest.raises(ToolInputError, match="Could not load tokenizer"):
            ChunkBySizeTool().run(
                {"corpus": "articles", "tokenizer": "no/such-model"}, context=ToolContext()
            )


class TestContextualize:
    def test_without_context_the_indexed_text_is_the_window(self):
        out = ChunkBySizeTool().run({"corpus": "articles", "size": 4}, context=ToolContext())
        chunk = get_corpus(out["corpus"]).get(["a1#1"])[0]
        assert chunk["text"] == chunk["chunk_text"] and chunk["context"] is None

    def test_the_title_is_prepended_before_indexing(self):
        out = ChunkBySizeTool().run(
            {"corpus": "articles", "size": 4, "contextualize": "title"}, context=ToolContext()
        )
        chunk = get_corpus(out["corpus"]).get(["a1#1"])[0]
        assert chunk["context"] == "Agents"
        assert chunk["text"] == "Agents\n\n" + chunk["chunk_text"]
        assert chunk["tokens"] == 4, "token counts refer to the window, not the context"
        assert out["corpus"] == "articles_size4_overlap0_title"
        assert out["provenance"]["settings"]["contextualize"] == "title"

    def test_context_lets_retrieval_match_what_the_window_itself_does_not_say(self):
        plain = ChunkBySizeTool().run({"corpus": "articles", "size": 4}, context=ToolContext())
        titled = ChunkBySizeTool().run(
            {"corpus": "articles", "size": 4, "contextualize": "title"}, context=ToolContext()
        )
        query = {"query": "chips", "size": 5}
        assert (
            RetrievePassagesTool().run({**query, "input": plain}, context=ToolContext())["returned"]
            == 0
        )
        found = RetrievePassagesTool().run({**query, "input": titled}, context=ToolContext())[
            "passages"
        ]
        assert [p["id"] for p in found] == ["a2#0"]

    def test_a_registered_contextualizer_can_be_chosen_by_name(self):
        register_contextualizer(
            "title_and_date", lambda record, chunk: f"{record['title']} ({record['date']})"
        )
        out = ChunkBySizeTool().run(
            {"corpus": "articles", "size": 4, "contextualize": "title_and_date"},
            context=ToolContext(),
        )
        assert get_corpus(out["corpus"]).get(["a1#0"])[0]["context"] == "Agents (2025-05-19)"

    def test_an_empty_context_leaves_the_window_untouched(self):
        register_contextualizer("nothing", lambda record, chunk: "  ")
        out = ChunkBySizeTool().run(
            {"corpus": "articles", "size": 4, "contextualize": "nothing"}, context=ToolContext()
        )
        chunk = get_corpus(out["corpus"]).get(["a1#0"])[0]
        assert chunk["context"] is None and chunk["text"] == chunk["chunk_text"]

    def test_an_unknown_contextualizer_is_explained(self):
        with pytest.raises(ToolInputError, match="contextualize must be one of"):
            ChunkBySizeTool().run(
                {"corpus": "articles", "contextualize": "summary"}, context=ToolContext()
            )

    def test_none_is_a_reserved_name(self):
        with pytest.raises(ValueError):
            register_contextualizer("none", lambda record, chunk: "x")
