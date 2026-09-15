"""chunk_by_size — fixed-size chunking, the baseline.

Every record's text is cut into windows of ``size`` tokens, optionally
overlapping by ``overlap`` tokens, without regard for paragraphs, sentences or
meaning. That is deliberate: this is the baseline that better chunking
strategies — structural, semantic, contextual — are measured against.

    chunk_by_size(input=articles, size=200, overlap=40)

**Tokens.** By default a token is a whitespace-separated word, which needs no
dependencies. Pass a Hugging Face tokenizer name — ideally the one of your
embedding model, e.g. ``sentence-transformers/all-MiniLM-L6-v2`` — to count that
model's tokens. Leave a few tokens of margin below the model's limit: the model adds
its own special tokens, and re-tokenizing a cut-out window can differ by a token at
its edges (a 256-token window measured 257 on the AI Media Dataset). That needs the
``embeddings`` extra.

**Overlap.** Consecutive windows share ``overlap`` tokens, so a statement cut
by one boundary appears whole in the next window. It costs more chunks for the
same text. The final window may be shorter than ``size``.

**Context (a blueprint).** A chunk cut out of an article no longer says what the
article is about — "the platform cut processing time by 30%": which platform?
``contextualize="title"`` prepends the document's title to every chunk before it is
indexed. That is deliberately the simplest form of contextual retrieval. Richer
context — date and source, a summary of the document, or a sentence a language model
writes for each chunk — is yours to add: write a function ``(record, chunk_text) -> str``
and register it with ``register_contextualizer``, or override ``contextualize`` in a
subclass. Each chunk keeps ``chunk_text`` (the window), ``context`` (what was added)
and ``text`` (both, which is what gets indexed); token counts and spans always refer
to the window.

**Other chunking methods.** Subclass ``ChunkBySizeTool``, give it a new
``tool_name`` and override ``split`` — see ``anatoolbox.tool`` and the course
notebook. Everything else (binding the input, metadata, context, the new corpus,
provenance) is inherited.

**Output.** A new corpus that ``retrieve_passages`` searches like any other.
Each chunk keeps its article id (``source_id``), its position (token and
character span) and the article's metadata, so a retrieved chunk can still be
cited, dated and traced back to its article.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable
from typing import Any

from anatoolbox.base import ToolContext
from anatoolbox.corpus import LocalCorpus, bind_corpus, local_ref, register_corpus
from anatoolbox.preprocess.chunk.base import PREFIX, STAGE
from anatoolbox.tool import BaseTool

TOOL_NAME = "chunk_by_size"
OBJECT_TYPE = "corpus"
DEFAULT_SIZE = 200
DEFAULT_OVERLAP = 0
NO_CONTEXT = "none"
WHITESPACE = "whitespace"

#: Maps a text to the (start, end) character span of each of its tokens.
Tokenizer = Callable[[str], list[tuple[int, int]]]

_NON_SPACE = re.compile(r"\S+")
_HF_TOKENIZERS: dict[str, Any] = {}
#: Chunk fields that describe the window, not the record it came from.
_CHUNK_OWN_FIELDS = (
    "id",
    "source_id",
    "chunk_index",
    "chunk_count",
    "context",
    "chunk_text",
    "text",
)

#: Writes context for one chunk, from its source record and the chunk's own text.
Contextualizer = Callable[[dict[str, Any], str], str]


def title_context(record: dict[str, Any], chunk_text: str) -> str:
    """The blueprint contextualizer: the document's title.

    Replace or extend it — add the date and source, a one-line summary of the
    document, or context a language model writes for this particular chunk.
    """
    return str(record.get("title") or "").strip()


_CONTEXTUALIZERS: dict[str, Contextualizer | None] = {NO_CONTEXT: None, "title": title_context}


def register_contextualizer(name: str, contextualizer: Contextualizer) -> None:
    """Make ``contextualize=name`` available to chunk_by_size."""
    if not name or name == NO_CONTEXT:
        raise ValueError(f"A contextualizer needs a name other than {NO_CONTEXT!r}.")
    _CONTEXTUALIZERS[name] = contextualizer


def contextualizer_names() -> list[str]:
    return sorted(_CONTEXTUALIZERS)


def _with_context(context: str, chunk_text: str) -> dict[str, Any]:
    """``text`` is what gets indexed and shown to a model; ``chunk_text`` is the window itself."""
    context = (context or "").strip()
    return {
        "context": context or None,
        "chunk_text": chunk_text,
        "text": f"{context}\n\n{chunk_text}" if context else chunk_text,
    }


def whitespace_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of whitespace-separated words."""
    return [(match.start(), match.end()) for match in _NON_SPACE.finditer(text)]


def huggingface_spans(name: str) -> Tokenizer:
    """A Tokenizer backed by a Hugging Face tokenizer, loaded once per name."""

    def spans(text: str) -> list[tuple[int, int]]:
        tokenizer = _load_hf_tokenizer(name)
        encoded = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        return [(start, end) for start, end in encoded["offset_mapping"] if end > start]

    return spans


def _load_hf_tokenizer(name: str) -> Any:
    tokenizer = _HF_TOKENIZERS.get(name)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        _HF_TOKENIZERS[name] = tokenizer
    return tokenizer


def chunk_text_by_size(
    text: str,
    *,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    tokenizer: Tokenizer = whitespace_spans,
) -> list[dict[str, Any]]:
    """Cut ``text`` into windows of ``size`` tokens overlapping by ``overlap``.

    Window text is sliced from the original, so spacing and line breaks inside a
    window are kept. Returns dicts with ``text``, ``tokens``, ``token_start``,
    ``token_end`` (exclusive), ``char_start`` and ``char_end``.
    """
    if size < 1 or not 0 <= overlap < size:
        raise ValueError(
            f"Need size >= 1 and 0 <= overlap < size, got size={size}, overlap={overlap}."
        )
    spans = tokenizer(text or "")
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(spans):
        end = min(start + size, len(spans))
        char_start, char_end = spans[start][0], spans[end - 1][1]
        chunks.append(
            {
                "text": text[char_start:char_end],
                "tokens": end - start,
                "token_start": start,
                "token_end": end,
                "char_start": char_start,
                "char_end": char_end,
            }
        )
        if end == len(spans):
            break
        start += size - overlap
    return chunks


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": "The corpus to chunk: a handle such as 'corpus_1'. Omit to use the most recent.",
        },
        "corpus": {
            "type": "string",
            "description": "Name of a registered corpus. Takes precedence over `input`.",
        },
        "name": {
            "type": "string",
            "description": "Name for the chunk corpus (default: '<source>_size<size>_overlap<overlap>').",
        },
        "size": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_SIZE,
            "description": "Tokens per chunk.",
        },
        "overlap": {
            "type": "integer",
            "minimum": 0,
            "default": DEFAULT_OVERLAP,
            "description": "Tokens shared by consecutive chunks; must be smaller than size.",
        },
        "tokenizer": {
            "type": "string",
            "default": WHITESPACE,
            "description": (
                "'whitespace' counts words; a Hugging Face tokenizer name (e.g. your embedding "
                "model's) counts that model's tokens."
            ),
        },
        "contextualize": {
            "type": "string",
            "default": NO_CONTEXT,
            "description": (
                "Context prepended to each chunk before indexing: 'none', 'title' (the document's "
                "title), or the name of a registered contextualizer."
            ),
        },
    },
}


class ChunkBySizeTool(BaseTool):
    """Fixed-size chunking with optional overlap — the baseline to beat.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``split(text, settings)`` — cut one record's text into chunks.
    * ``contextualize(record, chunk_text, settings)`` — context prepended before indexing.
    * ``settings(args)`` — read and check arguments; add your own here.
    * ``corpus_name(source, settings)`` — the name of the chunk corpus.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Split every record of a corpus into chunks of a fixed number of tokens, optionally "
        "overlapping, producing a new corpus that retrieve_passages can search. Each chunk keeps "
        "its article id, position and the article's metadata."
    )
    input_schema = _INPUT_SCHEMA

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments that shape the chunks. Recorded in provenance."""
        size = self.int_arg(args, "size", DEFAULT_SIZE, minimum=1)
        overlap = self.int_arg(args, "overlap", DEFAULT_OVERLAP, minimum=0)
        if overlap >= size:
            raise self.input_error(
                f"overlap must be smaller than size, got overlap={overlap}, size={size}.",
                size=size,
                overlap=overlap,
            )
        tokenizer = self.text_arg(args, "tokenizer") or WHITESPACE
        self.load_tokenizer(tokenizer)  # fail on a bad name before any chunking
        contextualize = self.text_arg(args, "contextualize") or NO_CONTEXT
        if contextualize not in _CONTEXTUALIZERS:
            raise self.input_error(
                f"contextualize must be one of {contextualizer_names()}, got {contextualize!r}.",
                argument="contextualize",
                value=contextualize,
            )
        return {
            "size": size,
            "overlap": overlap,
            "tokenizer": tokenizer,
            "contextualize": contextualize,
        }

    def split(self, text: str, settings: dict[str, Any]) -> list[dict[str, Any]]:
        """Cut one record's text into chunks, in order. Override to chunk differently.

        Return a list of dicts, each with the chunk's ``text``. Every other key is
        kept on the chunk — here ``tokens`` and the token and character spans. A
        chunk without ``tokens`` is counted in whitespace words.
        """
        return chunk_text_by_size(
            text,
            size=settings["size"],
            overlap=settings["overlap"],
            tokenizer=self.load_tokenizer(settings["tokenizer"]),
        )

    def contextualize(
        self, record: dict[str, Any], chunk_text: str, settings: dict[str, Any]
    ) -> str:
        """Context to prepend to one chunk ("" for none). Default: the registered contextualizer."""
        contextualizer = _CONTEXTUALIZERS[settings["contextualize"]]
        return contextualizer(record, chunk_text) if contextualizer else ""

    def corpus_name(self, source: LocalCorpus, settings: dict[str, Any]) -> str:
        """Default name of the chunk corpus, used unless the caller passes ``name``."""
        parts = [source.name]
        if self.tool_name != TOOL_NAME:
            # A variant: keep its corpus apart from the baseline's, which would
            # otherwise be silently replaced under the same name.
            parts.append(self.tool_name.removeprefix(PREFIX))
        if "size" in settings:
            parts.append(f"size{settings['size']}")
        if "overlap" in settings:
            parts.append(f"overlap{settings['overlap']}")
        if settings.get("contextualize", NO_CONTEXT) != NO_CONTEXT:
            parts.append(settings["contextualize"])
        return "_".join(parts)

    def load_tokenizer(self, name: str) -> Tokenizer:
        """``'whitespace'`` or a Hugging Face tokenizer name -> a Tokenizer."""
        if name == WHITESPACE:
            return whitespace_spans
        try:
            _load_hf_tokenizer(name)
        except ImportError as exc:
            raise self.input_error(
                f"tokenizer={name!r} needs a Hugging Face tokenizer: pip install 'anatoolbox[embeddings]', "
                "or use tokenizer='whitespace'.",
                code="missing_dependency",
                argument="tokenizer",
                value=name,
            ) from exc
        except Exception as exc:  # unknown model name, no network, ...
            raise self.input_error(
                f"Could not load tokenizer {name!r}: {exc}", argument="tokenizer", value=name
            ) from exc
        return huggingface_spans(name)

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        source, source_ref = bind_corpus(args, context, tool_name=self.tool_name)
        source_handle = None if isinstance(args.get("input"), dict) else source_ref
        name = self.text_arg(args, "name") or self.corpus_name(source, settings)
        if name == source.name:
            raise self.input_error(
                "The chunk corpus needs a different name from its source corpus.",
                argument="name",
                value=name,
            )

        rows: list[dict[str, Any]] = []
        chunked = empty = 0
        for record in source.records:
            pieces = [
                p
                for p in self.split(source.text_of(record), settings)
                if str(p.get("text") or "").strip()
            ]
            if not pieces:
                empty += 1
                continue
            chunked += 1
            source_id = str(record.get(source.id_field))
            metadata = {
                k: v for k, v in record.items() if k not in (source.text_field, source.id_field)
            }
            for index, piece in enumerate(pieces):
                chunk_text = str(piece["text"])
                rows.append(
                    {
                        **metadata,
                        **{k: v for k, v in piece.items() if k not in _CHUNK_OWN_FIELDS},
                        "tokens": piece.get("tokens", len(whitespace_spans(chunk_text))),
                        "id": f"{source_id}#{index}",
                        "source_id": source_id,
                        "chunk_index": index,
                        "chunk_count": len(pieces),
                        **_with_context(
                            self.contextualize(record, chunk_text, settings), chunk_text
                        ),
                    }
                )
        if not rows:
            raise self.input_error(
                f"No record in corpus {source.name!r} has text in {source.text_field!r}.",
                code="nothing_to_chunk",
                corpus=source.name,
                text_field=source.text_field,
            )

        chunks = register_corpus(
            LocalCorpus.from_records(
                rows,
                name=name,
                id_field="id",
                text_field="text",
                source=f"{self.tool_name}({source.name})",
                parse_lists=False,
            )
        )
        sizes = [row["tokens"] for row in rows]
        return {
            "corpus": chunks.name,
            "source_corpus": source.name,
            "input_handle": source_handle,
            "handle": self._remember(context, chunks, source, source_handle, settings),
            "articles": chunked,
            "empty_articles": empty,
            "chunks": len(rows),
            "tokens": {"min": min(sizes), "median": statistics.median(sizes), "max": max(sizes)},
            "settings": settings,
            "provenance": self.provenance(
                {"source_corpus": source.name, "corpus": chunks.name, **settings},
                derived_from=[source_ref],
            ),
        }

    def _remember(self, context, chunks, source, source_handle, settings) -> str | None:
        """Chunks by reference to their registered corpus, with lineage to the source."""
        if context.recordsets is None:
            return None
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=self.stage,
            produced_by=self.tool_name,
            args={"corpus": chunks.name, "source_corpus": source.name, **settings},
            ref=local_ref(corpus=chunks.name, ids=chunks.ids),
            count=len(chunks),
            summary=f"{len(chunks)} chunks from {source.name} ({self.tool_name})",
            derived_from=[source_handle] if source_handle else [],
        )
        return record.handle
