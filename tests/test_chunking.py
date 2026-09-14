"""chunk_articles_by_paragraph: the packing algorithm and the tool around it."""

import pytest

from anatoolbox import ToolContext
from anatoolbox.corpus import (
    LocalCorpus,
    clear_corpora,
    configure_embedder,
    get_corpus,
    local_ref,
    register_corpus,
)
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool
from anatoolbox.memory.facade import Memory
from anatoolbox.memory.policy import DefaultMemoryPolicy
from anatoolbox.memory.store import InMemoryRecordsetStore
from anatoolbox.preprocess.chunk.chunk_articles_by_paragraph import (
    ChunkArticlesByParagraphTool,
    chunk_paragraphs,
    split_paragraphs,
)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestChunkParagraphs:
    def test_packs_paragraphs_up_to_the_target(self):
        chunks = chunk_paragraphs(
            [words(40), words(40), words(40), words(40)], target_size=100, max_size=200, min_size=10
        )
        assert [c["words"] for c in chunks] == [80, 80]
        assert [(c["paragraph_start"], c["paragraph_end"]) for c in chunks] == [(0, 1), (2, 3)]

    def test_keeps_paragraph_boundaries_in_the_text(self):
        chunks = chunk_paragraphs(
            ["alpha beta", "gamma delta"], target_size=10, max_size=20, min_size=1
        )
        assert chunks[0]["text"] == "alpha beta\n\ngamma delta"

    def test_long_paragraphs_split_at_sentences_and_never_exceed_max(self):
        long_paragraph = ". ".join(words(30, f"s{i}_") for i in range(10)) + "."
        chunks = chunk_paragraphs(
            [long_paragraph, words(50)], target_size=60, max_size=80, min_size=5
        )
        assert all(c["words"] <= 80 for c in chunks)
        assert sum(c["words"] for c in chunks) == 350, "no words lost or duplicated"

    def test_a_short_heading_stays_with_the_text_it_introduces(self):
        """Measured on the AI Media Dataset: 603 headings became orphan chunks before this rule."""
        chunks = chunk_paragraphs(
            ["In this article", words(200), words(200)], target_size=150, max_size=300, min_size=30
        )
        assert [c["words"] for c in chunks] == [203, 200]
        assert chunks[0]["text"].startswith("In this article\n\n")

    def test_max_size_still_wins_over_keeping_a_heading_attached(self):
        chunks = chunk_paragraphs(
            ["a short heading", words(299)], target_size=150, max_size=300, min_size=30
        )
        assert [c["words"] for c in chunks] == [3, 299]

    def test_a_single_oversized_sentence_is_cut_by_words(self):
        chunks = chunk_paragraphs([words(250)], target_size=100, max_size=100, min_size=10)
        assert [c["words"] for c in chunks] == [100, 100, 50]

    def test_a_tiny_trailing_chunk_joins_its_predecessor(self):
        chunks = chunk_paragraphs([words(90), words(5)], target_size=90, max_size=120, min_size=20)
        assert [c["words"] for c in chunks] == [95]

    def test_trailing_merge_never_exceeds_max(self):
        chunks = chunk_paragraphs(
            [words(100), words(5)], target_size=100, max_size=100, min_size=20
        )
        assert [c["words"] for c in chunks] == [100, 5]

    def test_overlap_repeats_trailing_paragraphs(self):
        paragraphs = [words(50, "a"), words(50, "b"), words(50, "c")]
        chunks = chunk_paragraphs(paragraphs, target_size=100, max_size=200, min_size=1, overlap=1)
        assert [(c["paragraph_start"], c["paragraph_end"]) for c in chunks] == [(0, 1), (1, 2)]

    def test_overlap_never_pushes_a_chunk_past_max(self):
        chunks = chunk_paragraphs(
            [words(60), words(60), words(60)], target_size=60, max_size=100, min_size=1, overlap=1
        )
        assert all(c["words"] <= 100 for c in chunks)

    def test_a_custom_measure_sizes_the_chunks(self):
        chars = chunk_paragraphs(
            ["aaaa", "bbbb", "cccc"], target_size=8, max_size=8, min_size=1, measure=len
        )
        assert [c["words"] for c in chars] == [8, 4]

    def test_empty_input_gives_no_chunks(self):
        assert chunk_paragraphs([]) == []
        assert chunk_paragraphs(["", "   "]) == []

    @pytest.mark.parametrize(
        "sizes",
        [
            {"min_size": 50, "target_size": 10, "max_size": 100},
            {"min_size": 5, "target_size": 200, "max_size": 100},
            {"min_size": 0, "target_size": 10, "max_size": 100},
            {"min_size": 5, "target_size": 10, "max_size": 100, "overlap": -1},
        ],
    )
    def test_inconsistent_sizes_are_rejected(self, sizes):
        with pytest.raises(ValueError):
            chunk_paragraphs(["text"], **sizes)

    def test_split_paragraphs_accepts_lists_and_strings(self):
        assert split_paragraphs(["a", " ", "b "]) == ["a", "b"]
        assert split_paragraphs("a\n\n  \nb\nc") == ["a", "b\nc"]
        assert split_paragraphs(None) == []


ARTICLES = [
    {
        "id": "a1",
        "title": "Agents on the web",
        "date": "2025-05-19",
        "domain": "example",
        "content": [
            "Agents browse web services.",
            "A second paragraph about MCP servers and tools.",
            "Closing remarks.",
        ],
    },
    {
        "id": "a2",
        "title": "AI chips",
        "date": "2024-11-02",
        "domain": "example",
        "content": ["Accelerators remain scarce."],
    },
    {"id": "a3", "title": "Empty", "date": "2025-01-01", "domain": "example", "content": []},
]
SMALL = {"target_words": 6, "max_words": 12, "min_words": 1}


@pytest.fixture(autouse=True)
def clean():
    clear_corpora()
    configure_embedder(None)
    yield
    clear_corpora()


@pytest.fixture
def ctx():
    memory = Memory(InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="p", session_id="s")
    corpus = register_corpus(
        LocalCorpus.from_records(ARTICLES, name="articles", text_field="content")
    )
    memory.remember(
        object_type="corpus",
        stage="gather",
        produced_by="ingest_corpus",
        args={"corpus": "articles"},
        ref=local_ref(corpus="articles", ids=corpus.ids),
        count=len(corpus),
        summary="3 articles",
    )
    return ToolContext(project="p", session_id="s", recordsets=memory)


class TestChunkTool:
    def test_creates_a_chunk_corpus_derived_from_its_source(self, ctx):
        out = ChunkArticlesByParagraphTool().run(dict(SMALL), context=ctx)
        assert out["input_handle"] == "corpus_1" and out["handle"] == "corpus_2"
        assert out["corpus"] == "articles_paragraph_chunks"
        assert ctx.recordsets.get("corpus_2").derived_from == ["corpus_1"]
        assert (out["articles"], out["empty_articles"], out["chunks"]) == (2, 1, 4)

    def test_chunks_carry_ids_positions_and_metadata(self, ctx):
        ChunkArticlesByParagraphTool().run(dict(SMALL), context=ctx)
        chunks = get_corpus("articles_paragraph_chunks")
        first = chunks.get(["a1#0"])[0]
        assert (first["source_id"], first["chunk_index"], first["chunk_count"]) == ("a1", 0, 3)
        assert (first["title"], first["date"]) == ("Agents on the web", "2025-05-19")
        assert first["text"] == "Agents browse web services."
        assert "content" not in first, "the source text field is not duplicated into chunks"
        assert chunks.get(["a1#1"])[0]["paragraph_start"] == 1

    def test_chunks_are_stored_by_reference(self, ctx):
        out = ChunkArticlesByParagraphTool().run(dict(SMALL), context=ctx)
        ref = ctx.recordsets.get(out["handle"]).ref
        assert ref["mode"] == "reference" and "records" not in ref

    def test_context_header_prepends_title_and_date(self, ctx):
        ChunkArticlesByParagraphTool().run({**SMALL, "context_header": True}, context=ctx)
        text = get_corpus("articles_paragraph_chunks").get(["a1#1"])[0]["text"]
        assert (
            text
            == "Agents on the web — 2025-05-19\n\nA second paragraph about MCP servers and tools."
        )

    def test_retrieval_binds_and_searches_the_chunks(self, ctx):
        ChunkArticlesByParagraphTool().run(dict(SMALL), context=ctx)
        out = RetrievePassagesTool().run({"query": "MCP servers"}, context=ctx)
        assert out["input_handle"] == "corpus_2"
        assert out["passages"][0]["id"] == "a1#1"
        assert out["passages"][0]["source_id"] == "a1"

    def test_string_text_is_split_at_blank_lines(self, ctx):
        register_corpus(
            LocalCorpus.from_records([{"id": "s1", "text": "one two\n\nthree four"}], name="plain")
        )
        out = ChunkArticlesByParagraphTool().run(
            {"corpus": "plain", "target_words": 2, "max_words": 4, "min_words": 1}, context=ctx
        )
        assert out["chunks"] == 2 and out["input_handle"] is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"target_words": 0},
            {"min_words": 50, "target_words": 10},
            {"target_words": 500, "max_words": 100},
            {"overlap_paragraphs": -1},
            {"target_words": "many"},
            {"target_words": True},
        ],
    )
    def test_invalid_sizes_are_tool_input_errors(self, ctx, bad):
        with pytest.raises(ToolInputError):
            ChunkArticlesByParagraphTool().run(bad, context=ctx)

    def test_output_name_must_differ_from_the_source(self, ctx):
        with pytest.raises(ToolInputError, match="different name"):
            ChunkArticlesByParagraphTool().run({**SMALL, "name": "articles"}, context=ctx)

    def test_unknown_corpus_is_a_tool_input_error(self, ctx):
        with pytest.raises(ToolInputError, match="No corpus named"):
            ChunkArticlesByParagraphTool().run({**SMALL, "corpus": "nope"}, context=ctx)

    def test_a_corpus_without_text_is_explained(self, ctx):
        register_corpus(LocalCorpus.from_records([{"id": "e", "text": ""}], name="blank"))
        with pytest.raises(ToolInputError, match="has text"):
            ChunkArticlesByParagraphTool().run({**SMALL, "corpus": "blank"}, context=ctx)
