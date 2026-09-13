import json

import pytest

from anatoolbox.corpus import (
    LocalCorpus,
    clear_corpora,
    configure_embedder,
    corpus_names,
    ensure_bm25,
    ensure_embeddings,
    get_corpus,
    local_ref,
    parse_list_string,
    register_corpus,
    resolve_local_ref,
)

ROWS = [
    {"id": "d1", "title": "Agentic web", "content": "autonomous agents and web standards"},
    {"id": "d2", "title": "AI chips", "content": "accelerators and data centers"},
]


@pytest.fixture(autouse=True)
def clean_corpora():
    clear_corpora()
    configure_embedder(None)
    yield
    clear_corpora()
    configure_embedder(None)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_from_records_indexes_by_id():
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    assert len(corpus) == 2
    assert corpus.ids == ["d1", "d2"]
    assert corpus.get(["d2"])[0]["title"] == "AI chips"


def test_get_preserves_requested_order_and_skips_missing():
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    got = corpus.get(["d2", "nope", "d1"])
    assert [r["id"] for r in got] == ["d2", "d1"]


def test_missing_ids_are_synthesized_not_fatal():
    corpus = LocalCorpus.from_records(
        [{"content": "a"}, {"id": "", "content": "b"}], name="c", text_field="content"
    )
    assert corpus.ids == ["c-0", "c-1"]


def test_text_of_joins_paragraph_lists():
    """Some exports store text as a list of paragraphs rather than a blob."""
    corpus = LocalCorpus.from_records(
        [{"id": "d1", "content": ["para one", "para two"]}], name="c", text_field="content"
    )
    assert corpus.texts() == ["para one\n\npara two"], "paragraph boundaries must survive"


def test_text_of_handles_missing_value():
    corpus = LocalCorpus.from_records([{"id": "d1"}], name="c", text_field="content")
    assert corpus.texts() == [""]


def test_csv_round_trip(tmp_path):
    path = write(tmp_path, "c.csv", "id,content\nd1,hello world\nd2,second doc\n")
    corpus = LocalCorpus.from_file(path, text_field="content")
    assert len(corpus) == 2
    assert corpus.name == "c"
    assert corpus.source == str(path)


def test_jsonl_round_trip(tmp_path):
    body = "\n".join(json.dumps(r) for r in ROWS)
    corpus = LocalCorpus.from_file(write(tmp_path, "c.jsonl", body), text_field="content")
    assert corpus.ids == ["d1", "d2"]


def test_json_accepts_a_wrapper_object(tmp_path):
    path = write(tmp_path, "c.json", json.dumps({"records": ROWS}))
    assert len(LocalCorpus.from_file(path, text_field="content")) == 2


def test_limit_truncates(tmp_path):
    path = write(tmp_path, "c.csv", "id,content\nd1,a\nd2,b\nd3,c\n")
    assert len(LocalCorpus.from_file(path, text_field="content", limit=2)) == 2


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalCorpus.from_file(tmp_path / "nope.csv")


def test_unsupported_format_is_reported(tmp_path):
    with pytest.raises(ValueError, match="Unsupported corpus format"):
        LocalCorpus.from_file(write(tmp_path, "c.xml", "<x/>"))


def test_wrong_text_field_names_the_columns(tmp_path):
    path = write(tmp_path, "c.csv", "id,body\nd1,hello\n")
    with pytest.raises(ValueError, match="Columns:"):
        LocalCorpus.from_file(path, text_field="content")


def test_empty_file_is_reported(tmp_path):
    with pytest.raises(ValueError, match="no records"):
        LocalCorpus.from_file(write(tmp_path, "c.csv", "id,content\n"), text_field="content")


def test_registry_round_trip():
    corpus = register_corpus(LocalCorpus.from_records(ROWS, name="c", text_field="content"))
    assert corpus_names() == ["c"]
    assert get_corpus("c") is corpus


def test_unknown_corpus_lists_what_is_registered():
    with pytest.raises(KeyError, match="Registered:"):
        get_corpus("nope")


def test_local_ref_resolves_to_records():
    register_corpus(LocalCorpus.from_records(ROWS, name="c", text_field="content"))
    records, missing = resolve_local_ref(local_ref(corpus="c", ids=["d1"]))
    assert [r["id"] for r in records] == ["d1"]
    assert missing == 0


def test_local_ref_reports_missing_ids():
    register_corpus(LocalCorpus.from_records(ROWS, name="c", text_field="content"))
    records, missing = resolve_local_ref(local_ref(corpus="c", ids=["d1", "gone"]))
    assert len(records) == 1 and missing == 1


def test_local_ref_honours_source_includes():
    """Projection keeps the id even when it is not requested."""
    register_corpus(LocalCorpus.from_records(ROWS, name="c", text_field="content"))
    records, _ = resolve_local_ref(local_ref(corpus="c", ids=["d1"]), ["title"])
    assert records[0] == {"id": "d1", "title": "Agentic web"}


def test_embeddings_are_computed_once():
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    calls = []

    def embedder(texts):
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    configure_embedder(embedder)
    first = ensure_embeddings(corpus)
    second = ensure_embeddings(corpus)
    assert first is second
    assert calls == [2], "embeddings should be cached on the corpus"


def test_list_literal_cells_become_lists(tmp_path):
    """Tabular exports stringify paragraph and tag lists; undo that on load."""
    path = write(
        tmp_path,
        "c.csv",
        "id,content,tags\nd1,\"['para one', 'para two']\",\"['Robotics', 'GPU']\"\n",
    )
    corpus = LocalCorpus.from_file(path, text_field="content")
    record = corpus.get(["d1"])[0]
    assert record["tags"] == ["Robotics", "GPU"]
    assert corpus.texts() == ["para one\n\npara two"]


def test_list_parsing_can_be_disabled(tmp_path):
    path = write(tmp_path, "c.csv", "id,content\nd1,\"['a', 'b']\"\n")
    corpus = LocalCorpus.from_file(path, text_field="content", parse_lists=False)
    assert corpus.texts() == ["['a', 'b']"]


@pytest.mark.parametrize(
    "value",
    ["[Video] New model released", "[1, 2", "not a list", "", "[", "{'a': 1}", 42, None],
)
def test_parse_list_string_leaves_non_lists_alone(value):
    assert parse_list_string(value) == value


def test_parse_list_string_does_not_turn_dicts_into_lists():
    assert parse_list_string("{'a': 1}") == "{'a': 1}"


def test_texts_are_cached_and_refresh_clears_them():
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    assert corpus.texts() is corpus.texts()
    corpus.records.append({"id": "d3", "content": "new"})
    corpus.refresh()
    assert len(corpus.texts()) == 3
    assert corpus.get(["d3"])


def test_bm25_index_is_built_once():
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    assert ensure_bm25(corpus) is ensure_bm25(corpus)


def test_switching_the_configured_embedder_recomputes_embeddings():
    """Stage 2 -> Stage 3: plugging in a new model must not reuse old vectors."""
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    configure_embedder(lambda texts: [[1.0, 0.0] for _ in texts])
    first = ensure_embeddings(corpus)
    configure_embedder(lambda texts: [[0.0, 1.0, 0.0] for _ in texts])
    second = ensure_embeddings(corpus)
    assert first != second
    assert len(second[0]) == 3
    assert ensure_embeddings(corpus) is second, "same model -> cached"


def test_explicit_embedder_is_cached_by_identity():
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    calls = []

    def embedder(texts):
        calls.append(1)
        return [[1.0] for _ in texts]

    ensure_embeddings(corpus, embedder)
    ensure_embeddings(corpus, embedder)
    assert calls == [1]
    ensure_embeddings(corpus, lambda texts: [[2.0] for _ in texts])
    assert corpus._embeddings == [[2.0], [2.0]]


def test_refresh_drops_embeddings():
    corpus = LocalCorpus.from_records(ROWS, name="c", text_field="content")
    configure_embedder(lambda texts: [[1.0] for _ in texts])
    ensure_embeddings(corpus)
    corpus.refresh()
    assert corpus._embeddings is None and corpus._embeddings_key is None
