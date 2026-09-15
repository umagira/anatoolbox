"""Local corpora — run the toolbox on a file, with no external services.

A ``LocalCorpus`` holds records in memory and hands them back by id. That is
all a reference-mode recordset needs, so registering the ``"local"`` resolver
(done at import time, below) is enough to make the whole recordset dataflow
work against a CSV on disk:

    corpus = LocalCorpus.from_file("ai_media.csv", text_field="content")
    register_corpus(corpus)

    ref = reference_ref(store="local", corpus=corpus.name, ids=["d1", "d2"])
    # ... memory.records(recordset) now returns those two rows

Reading is deliberately stdlib-only for CSV/JSON/JSONL, so the common case
needs no dependencies at all. Parquet needs ``pandas``/``pyarrow``, and dense
retrieval needs ``numpy`` — both in the ``local`` extra.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A CSV cell can legitimately be a whole article.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_ID_FIELD = "id"
DEFAULT_TEXT_FIELD = "text"
PARAGRAPH_SEPARATOR = "\n\n"


def parse_list_string(value: Any) -> Any:
    """``"['a', 'b']"`` -> ``['a', 'b']``; anything else comes back unchanged.

    Only a string that is *entirely* a valid list literal is converted, so a
    title like ``"[Video] New model released"`` is left alone.
    """
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] != "[" or stripped[-1] != "]":
        return value
    try:
        parsed = ast.literal_eval(stripped)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        return value
    return parsed if isinstance(parsed, list) else value


@dataclass
class LocalCorpus:
    """Records addressable by id, plus the field names that give them meaning.

    ``records`` are plain dicts — whatever columns the source file had. Only
    two are privileged: ``id_field`` (how a recordset refers to a row) and
    ``text_field`` (what retrieval searches).
    """

    name: str
    records: list[dict[str, Any]]
    id_field: str = DEFAULT_ID_FIELD
    text_field: str = DEFAULT_TEXT_FIELD
    source: str = ""
    _by_id: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    #: Cached embedding matrix, populated by ensure_embeddings().
    _embeddings: Any = field(default=None, repr=False, compare=False)
    #: Which embedder produced ``_embeddings``; a different one invalidates them.
    _embeddings_key: Any = field(default=None, repr=False, compare=False)
    #: Cached joined texts and BM25 index. Built on first use, cleared by refresh().
    _texts: Any = field(default=None, repr=False, compare=False)
    _bm25: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._reindex()

    def refresh(self) -> None:
        """Rebuild the id index and drop cached texts, BM25 and embeddings.

        Call after mutating ``records`` in place; everything derived from the
        old records would otherwise silently describe data that is gone.
        """
        self._texts = None
        self._bm25 = None
        self._embeddings = None
        self._embeddings_key = None
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {}
        for position, record in enumerate(self.records):
            key = record.get(self.id_field)
            if key is None or str(key).strip() == "":
                # Synthesize a stable id so a file without one still works.
                key = f"{self.name}-{position}"
                record[self.id_field] = key
            self._by_id[str(key)] = record

    # --- construction ----------------------------------------------------

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        name: str,
        id_field: str = DEFAULT_ID_FIELD,
        text_field: str = DEFAULT_TEXT_FIELD,
        source: str = "",
        parse_lists: bool = True,
    ) -> LocalCorpus:
        """Build a corpus from dicts.

        ``parse_lists`` turns string cells that hold a Python/JSON list literal
        (``"['para one', 'para two']"``) back into lists. Tabular exports do
        this to paragraph and tag columns constantly; left alone, every snippet
        and every prompt starts with ``['``.
        """
        rows = [dict(r) for r in records]
        if parse_lists:
            rows = [{k: parse_list_string(v) for k, v in r.items()} for r in rows]
        return cls(
            name=name,
            records=rows,
            id_field=id_field,
            text_field=text_field,
            source=source,
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        id_field: str = DEFAULT_ID_FIELD,
        text_field: str = DEFAULT_TEXT_FIELD,
        limit: int | None = None,
        parse_lists: bool = True,
    ) -> LocalCorpus:
        """Load ``.csv`` / ``.tsv`` / ``.json`` / ``.jsonl`` / ``.parquet``."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such corpus file: {path}")
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            records = _read_delimited(
                path, delimiter="\t" if suffix == ".tsv" else ",", limit=limit
            )
        elif suffix == ".jsonl":
            records = _read_jsonl(path, limit=limit)
        elif suffix == ".json":
            records = _read_json(path, limit=limit)
        elif suffix in (".parquet", ".pq"):
            records = _read_parquet(path, limit=limit)
        else:
            raise ValueError(
                f"Unsupported corpus format {suffix!r}. Use .csv, .tsv, .json, .jsonl or .parquet."
            )
        if not records:
            raise ValueError(f"{path} contained no records.")
        missing_text = text_field not in records[0]
        if missing_text:
            raise ValueError(
                f"text_field {text_field!r} is not a column in {path.name}. "
                f"Columns: {sorted(records[0])}"
            )
        return cls.from_records(
            records,
            name=name or path.stem,
            id_field=id_field,
            text_field=text_field,
            source=str(path),
            parse_lists=parse_lists,
        )

    # --- access ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    @property
    def ids(self) -> list[str]:
        return list(self._by_id)

    def get(self, ids: Iterable[str]) -> list[dict[str, Any]]:
        """Records for ``ids``, in the order asked, skipping ones not present."""
        found = []
        for key in ids:
            record = self._by_id.get(str(key))
            if record is not None:
                found.append(record)
        return found

    def text_of(self, record: dict[str, Any]) -> str:
        value = record.get(self.text_field)
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            # A paragraph list. Keep the boundaries: they are what a chunker
            # splits on later, and they keep snippets readable.
            return PARAGRAPH_SEPARATOR.join(str(part) for part in value)
        return str(value)

    def texts(self) -> list[str]:
        """Searchable text for every record, computed once and cached."""
        if self._texts is None:
            self._texts = [self.text_of(r) for r in self.records]
        return self._texts

    def describe(self) -> dict[str, Any]:
        columns = sorted(self.records[0]) if self.records else []
        return {
            "name": self.name,
            "records": len(self.records),
            "id_field": self.id_field,
            "text_field": self.text_field,
            "columns": columns,
            "source": self.source,
        }


# --- file readers --------------------------------------------------------


def _truncate(rows: Iterable[dict], limit: int | None) -> list[dict]:
    out = []
    for row in rows:
        if limit is not None and len(out) >= limit:
            break
        out.append(row)
    return out


def _read_delimited(path: Path, *, delimiter: str, limit: int | None) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return _truncate(csv.DictReader(handle, delimiter=delimiter), limit)


def _read_jsonl(path: Path, *, limit: int | None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if limit is not None and len(rows) >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path, *, limit: int | None) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Tolerate {"records": [...]} / {"data": [...]} wrappers.
        for key in ("records", "data", "rows", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list of records.")
    return _truncate(data, limit)


def _read_parquet(path: Path, *, limit: int | None) -> list[dict]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "Reading parquet needs pandas + pyarrow: pip install 'anatoolbox[local]'"
        ) from exc
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_dict(orient="records")


# --- corpus registry -----------------------------------------------------
#
# Reference-mode recordsets store a corpus *name*, not the corpus itself, so
# that recordset metadata stays small and serializable. The name is resolved
# here at read time.

_CORPORA: dict[str, LocalCorpus] = {}


def register_corpus(corpus: LocalCorpus, *, replace: bool = True) -> LocalCorpus:
    """Make ``corpus`` resolvable by name. Returns it, for chaining."""
    if corpus.name in _CORPORA and not replace:
        raise ValueError(f"Corpus {corpus.name!r} is already registered.")
    _CORPORA[corpus.name] = corpus
    return corpus


def get_corpus(name: str) -> LocalCorpus:
    try:
        return _CORPORA[name]
    except KeyError:
        raise KeyError(
            f"No corpus named {name!r}. Registered: {sorted(_CORPORA)}. "
            "Load one with LocalCorpus.from_file(...) then register_corpus(...)."
        ) from None


def corpus_names() -> list[str]:
    return sorted(_CORPORA)


def clear_corpora() -> None:
    _CORPORA.clear()


# --- recordset integration -----------------------------------------------


def resolve_local_ref(ref: dict, source_includes: list | None = None) -> tuple[list[dict], int]:
    """Resolver for ``store="local"`` refs. See memory.register_ref_resolver."""
    ids = [str(i) for i in ref.get("ids") or []]
    if not ids:
        return [], 0
    corpus = get_corpus(str(ref.get("corpus")))
    records = corpus.get(ids)
    if source_includes:
        keep = set(source_includes) | {corpus.id_field}
        records = [{k: v for k, v in r.items() if k in keep} for r in records]
    return records, len(ids) - len(records)


def _register_resolver() -> None:
    from anatoolbox.memory import register_ref_resolver

    register_ref_resolver("local", resolve_local_ref)


_register_resolver()


def local_ref(*, corpus: str, ids: Iterable[str]) -> dict:
    """Reference-mode ref pointing at rows of a registered local corpus."""
    from anatoolbox.memory import reference_ref

    return reference_ref(store="local", corpus=str(corpus), ids=list(ids))


# --- embeddings ----------------------------------------------------------

#: Turns texts into vectors. Any callable will do — that is the seam that
#: lets a course plug in the model they fine-tuned themselves.
Embedder = Callable[[list[str]], Any]


def sentence_transformer_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """An Embedder backed by sentence-transformers (in the ``local`` extra)."""

    def embed(texts: list[str]) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "Dense retrieval needs sentence-transformers: pip install 'anatoolbox[local]'"
            ) from exc

        model = _load_model(model_name, SentenceTransformer)
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    return embed


_MODELS: dict[str, Any] = {}


def _load_model(model_name: str, factory: Any) -> Any:
    """Cache models by name — loading one costs seconds."""
    model = _MODELS.get(model_name)
    if model is None:
        model = factory(model_name)
        _MODELS[model_name] = model
    return model


_EMBEDDER: Embedder | None = None
#: Bumped by every configure_embedder() call, so embeddings cached under a
#: previous model are recognized as stale rather than silently reused.
_EMBEDDER_GENERATION = 0
_DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def configure_embedder(embedder: Embedder | None) -> None:
    """Set the process-wide Embedder used by dense retrieval.

    Tools receive JSON arguments, so a callable cannot arrive that way. A
    notebook injects its own model here — including one fine-tuned in a
    previous stage — and every dense retrieval picks it up. Pass ``None`` to
    fall back to the default sentence-transformers model.
    """
    global _EMBEDDER, _EMBEDDER_GENERATION
    _EMBEDDER = embedder
    _EMBEDDER_GENERATION += 1


def get_embedder() -> Embedder:
    """The configured Embedder, or a lazily-built default."""
    if _EMBEDDER is not None:
        return _EMBEDDER
    return sentence_transformer_embedder(_DEFAULT_EMBEDDING_MODEL)


def ensure_bm25(corpus: LocalCorpus) -> Any:
    """Build the corpus's BM25 index once and cache it.

    Rebuilding per query costs ~10 s on a 16k-article corpus; an evaluation
    loop over hundreds of questions cannot afford that.
    """
    if corpus._bm25 is None:
        from anatoolbox.retrieval import BM25Index

        corpus._bm25 = BM25Index(corpus.texts())
    return corpus._bm25


def _embedder_key(embedder: Embedder | None) -> tuple:
    if embedder is None:
        return ("configured", _EMBEDDER_GENERATION)
    return ("explicit", id(embedder))


def ensure_embeddings(corpus: LocalCorpus, embedder: Embedder | None = None) -> Any:
    """Embed a corpus once per embedder and cache the matrix on it.

    The cache is keyed by the embedder that produced it. Switching models —
    say, to one fine-tuned in an earlier stage — recomputes instead of reusing
    vectors from the old model, which would either fail on a dimension
    mismatch or, worse, silently rank with the wrong model.
    """
    key = _embedder_key(embedder)
    if corpus._embeddings is not None and corpus._embeddings_key == key:
        return corpus._embeddings
    matrix = (embedder or get_embedder())(corpus.texts())
    corpus._embeddings = matrix
    corpus._embeddings_key = key
    return matrix


def save_embeddings(corpus: LocalCorpus, path: str | Path) -> Path:
    """Persist a corpus's embedding matrix so a restart need not recompute it."""
    np = _require_numpy_for_corpus()
    matrix = getattr(corpus, "_embeddings", None)
    if matrix is None:
        raise ValueError(f"Corpus {corpus.name!r} has no embeddings yet.")
    path = Path(path)
    np.save(path, np.asarray(matrix))
    return path


def load_embeddings(corpus: LocalCorpus, path: str | Path) -> Any:
    """Attach a previously saved embedding matrix, checking it still fits."""
    np = _require_numpy_for_corpus()
    matrix = np.load(Path(path))
    if len(matrix) != len(corpus):
        raise ValueError(
            f"Embedding matrix has {len(matrix)} rows but corpus {corpus.name!r} "
            f"has {len(corpus)} records — it was built for a different corpus."
        )
    corpus._embeddings = matrix
    # Assumed to come from the currently configured embedder; that is the
    # model retrieval will encode queries with.
    corpus._embeddings_key = _embedder_key(None)
    return matrix


def _require_numpy_for_corpus() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(
            "Saving or loading embeddings needs numpy: pip install 'anatoolbox[local]'"
        ) from exc
    return np


# --- consumer binding ------------------------------------------------------


def bind_corpus(
    args: dict[str, Any], context: Any, *, tool_name: str
) -> tuple[LocalCorpus, str | None]:
    """Resolve which corpus a consuming tool works on, and what it came from.

    In precedence order:

    1. ``corpus`` — an explicit corpus name. Nothing upstream is recorded.
    2. ``input`` as a result object (pipelines and notebooks) — the result of
       ``ingest_corpus`` or ``chunk_by_size``, which names its
       ``corpus``. Returns that result's ``run_id``.
    3. ``input`` as a handle, or no input at all (agents with recordset memory)
       — binds the named recordset, or else the newest ``corpus`` recordset.
       Returns the handle.

    Without recordset memory nothing is bound implicitly: a pipeline names its input.
    """
    from anatoolbox.errors import ToolInputError

    named = str(args.get("corpus") or "").strip()
    if named:
        return _corpus_or_input_error(named, tool_name), None

    given = args.get("input")
    if isinstance(given, dict):
        name = str(given.get("corpus") or "").strip()
        if not name:
            raise ToolInputError(
                code="invalid_argument_value",
                message=(
                    "`input` must be a corpus handle or a result that names its corpus, "
                    "such as the result of ingest_corpus or chunk_by_size."
                ),
                tool_name=tool_name,
                details={"argument": "input"},
            )
        from anatoolbox.provenance import run_id_of

        return _corpus_or_input_error(name, tool_name), run_id_of(given)

    recordsets = getattr(context, "recordsets", None)
    if recordsets is None:
        raise ToolInputError(
            code="missing_required_arguments",
            message=(
                "No corpus given. Pass corpus='<name>', or input=<the result of ingest_corpus "
                "or chunk_by_size>; with recordset memory, a handle also works."
            ),
            tool_name=tool_name,
            details={"missing": ["corpus"]},
        )
    requested = given.strip() if isinstance(given, str) and given.strip() else None
    record = recordsets.bind(object_type="corpus", tool_name=tool_name, requested=requested)
    name = str(record.ref.get("corpus") or record.args.get("corpus") or "")
    return _corpus_or_input_error(name, tool_name), record.handle


def _corpus_or_input_error(name: str, tool_name: str) -> LocalCorpus:
    from anatoolbox.errors import ToolInputError

    try:
        return get_corpus(name)
    except KeyError as exc:
        raise ToolInputError(
            code="unknown_corpus",
            message=str(exc.args[0]) if exc.args else f"No corpus named {name!r}.",
            tool_name=tool_name,
            details={"corpus": name},
        ) from exc


def passages_with_text(
    passages: Any, corpus_name: str | None, *, tool_name: str
) -> list[dict[str, Any]]:
    """Passages as given, with full ``text`` looked up from their corpus where it is missing.

    Retrieval and reranking results carry short snippets rather than full texts,
    so results stay small. A tool that needs the full text is told the corpus the
    passages came from — in a pipeline, the upstream result's ``corpus`` field —
    and reads each text from it by id.
    """
    from anatoolbox.errors import ToolInputError

    usage = (
        "passages must be a list of objects with 'id' and 'text' — or with 'id' alone when "
        "their corpus is known (pass corpus='<name>' or input=<the result that produced them>)."
    )
    if not isinstance(passages, list) or not all(
        isinstance(p, dict) and "id" in p for p in passages
    ):
        raise ToolInputError(
            code="invalid_argument_value",
            message=usage,
            tool_name=tool_name,
            details={"argument": "passages"},
        )
    missing = [str(p["id"]) for p in passages if "text" not in p]
    if missing and not corpus_name:
        raise ToolInputError(
            code="invalid_argument_value",
            message=usage,
            tool_name=tool_name,
            details={"argument": "passages", "without_text": missing[:10]},
        )
    texts: dict[str, str] = {}
    if missing:
        corpus = _corpus_or_input_error(corpus_name, tool_name)
        texts = {str(row.get(corpus.id_field)): corpus.text_of(row) for row in corpus.get(missing)}
        unknown = [passage_id for passage_id in missing if passage_id not in texts]
        if unknown:
            raise ToolInputError(
                code="unknown_ids",
                message=f"{len(unknown)} passage id(s) are not in corpus {corpus_name!r}: {unknown[:5]}",
                tool_name=tool_name,
                details={"corpus": corpus_name, "unknown_ids": unknown[:20]},
            )
    return [
        {**p, "id": str(p["id"]), "text": str(p["text"]) if "text" in p else texts[str(p["id"])]}
        for p in passages
    ]
