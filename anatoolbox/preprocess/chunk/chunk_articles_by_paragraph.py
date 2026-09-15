"""chunk_articles_by_paragraph — split a corpus into paragraph-packed chunks.

A ``chunk_`` instantiation for RAG over article-like text. Articles are split
at paragraph boundaries and paragraphs are packed into chunks of a target
size, so each chunk tends to stay on one topic and is small enough for an
embedding model or a reranker to read in full.

The result is itself a corpus — every chunk a record that remembers which
article and paragraphs it came from and carries the article's metadata — so
``retrieve_passages`` binds it exactly like the original:

    chunk_articles_by_paragraph(target_words=150)   # corpus_1 -> corpus_2
    retrieve_passages(query="...")                  # searches corpus_2

Design choices, and the trade-offs behind them:

* **Structure first, size second.** Paragraph boundaries are the cheapest
  meaningful boundary a document offers. A paragraph longer than
  ``max_words`` is split at sentence boundaries; only a single sentence longer
  than that is cut by word count. A short paragraph, such as a heading, stays
  with the text that follows it instead of becoming a chunk of its own.
* **Chunk size is a trade-off, not a constant.** Small chunks match specific
  facts precisely but lose the surrounding meaning; large chunks keep context
  but dilute the match and crowd a prompt. Treat ``target_words`` as a
  parameter to evaluate, not a default to trust.
* **Chunks lose context.** A chunk saying "the platform cut processing time by
  30%" does not say which platform, or when. ``context_header`` prepends the
  article's title and date to each chunk — a cheap, model-free step towards
  contextual retrieval (Anthropic, 2024), where a language model writes a short
  situating summary for every chunk instead.
* **Lineage down to the paragraph.** Every chunk records its source article
  and paragraph span, so a retrieved chunk can be traced, cited, dated, and
  expanded back to its full article.

Sizes are counted in words. Subword tokenizers typically produce roughly
1.3-1.5 tokens per English word, so ``target_words=150`` is about 200 tokens.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from anatoolbox.base import ToolContext, ToolSchema
from anatoolbox.corpus import (
    PARAGRAPH_SEPARATOR,
    LocalCorpus,
    bind_corpus,
    local_ref,
    register_corpus,
)
from anatoolbox.errors import ToolInputError
from anatoolbox.preprocess.chunk.base import PREFIX, STAGE
from anatoolbox.provenance import make_provenance

TOOL_NAME = "chunk_articles_by_paragraph"
OBJECT_TYPE = "corpus"
DEFAULT_TARGET_WORDS = 150
DEFAULT_MAX_WORDS = 300
DEFAULT_MIN_WORDS = 30
#: Record fields joined into the optional context header, in order.
HEADER_FIELDS = ("title", "date")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_BLANK_LINE = re.compile(r"\n\s*\n")

#: Measures a piece of text. Words by default; pass a token counter to size
#: chunks for a specific model.
Measure = Callable[[str], int]


def word_count(text: str) -> int:
    return len(text.split())


def split_paragraphs(value: Any) -> list[str]:
    """A paragraph list as given; a string split at blank lines. Blank parts are dropped."""
    if value is None:
        return []
    parts = value if isinstance(value, (list, tuple)) else _BLANK_LINE.split(str(value))
    return [str(part).strip() for part in parts if str(part).strip()]


def _units(paragraphs: Sequence[str], max_size: int, measure: Measure) -> list[tuple]:
    """Break paragraphs into ``(unit_id, text, paragraph_index, size)`` no larger than max_size."""
    units: list[tuple] = []

    def add(text: str, paragraph_index: int) -> None:
        units.append((len(units), text, paragraph_index, measure(text)))

    for index, paragraph in enumerate(paragraphs):
        if measure(paragraph) <= max_size:
            add(paragraph, index)
            continue
        # Too long: fall back to sentences, and cut a single oversized sentence
        # by words (exact for the default word measure, approximate otherwise).
        pieces: list[str] = []
        for sentence in _SENTENCE_END.split(paragraph):
            if measure(sentence) <= max_size:
                pieces.append(sentence)
            else:
                words = sentence.split()
                pieces.extend(
                    " ".join(words[i : i + max_size]) for i in range(0, len(words), max_size)
                )
        current: list[str] = []
        size = 0
        for piece in pieces:
            piece_size = measure(piece)
            if current and size + piece_size > max_size:
                add(" ".join(current), index)
                current, size = [], 0
            current.append(piece)
            size += piece_size
        if current:
            add(" ".join(current), index)
    return units


def chunk_paragraphs(
    paragraphs: Any,
    *,
    target_size: int = DEFAULT_TARGET_WORDS,
    max_size: int = DEFAULT_MAX_WORDS,
    min_size: int = DEFAULT_MIN_WORDS,
    overlap: int = 0,
    measure: Measure = word_count,
) -> list[dict[str, Any]]:
    """Pack paragraphs into chunks. The pipeline-level API behind the tool.

    Paragraphs are added to a chunk until the next one would take it past
    ``target_size`` — unless the chunk is still below ``min_size``, so a short
    paragraph such as a heading is kept with the text it introduces. No chunk
    exceeds ``max_size``; that bound wins over keeping a heading attached. A trailing chunk smaller
    than ``min_size`` joins its predecessor when the result still fits.
    ``overlap`` repeats that many trailing paragraphs (or paragraph pieces) at
    the start of the next chunk, so a fact split across a boundary appears
    whole in at least one chunk.

    Returns dicts with ``text``, ``words`` (size by ``measure``),
    ``paragraph_start`` and ``paragraph_end`` (inclusive indices).
    """
    if not (0 < min_size <= target_size <= max_size):
        raise ValueError(
            f"Sizes must satisfy 0 < min_size <= target_size <= max_size, "
            f"got {min_size}, {target_size}, {max_size}."
        )
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}.")

    units = _units(split_paragraphs(paragraphs), max_size, measure)
    chunks: list[list[tuple]] = []
    current: list[tuple] = []
    size = 0
    for unit in units:
        over_max = size + unit[3] > max_size
        over_target = size + unit[3] > target_size and size >= min_size
        if current and (over_max or over_target):
            chunks.append(current)
            carried = current[-overlap:] if overlap else []
            if sum(u[3] for u in carried) + unit[3] > max_size:
                carried = []  # overlap must never push a chunk past max_size
            current = list(carried)
            size = sum(u[3] for u in current)
        current.append(unit)
        size += unit[3]
    if current:
        chunks.append(current)

    if len(chunks) > 1:
        tail, previous = chunks[-1], chunks[-2]
        seen = {u[0] for u in previous}
        fresh = [u for u in tail if u[0] not in seen]
        tail_size = sum(u[3] for u in tail)
        merged_size = sum(u[3] for u in previous) + sum(u[3] for u in fresh)
        if tail_size < min_size and merged_size <= max_size:
            chunks[-2] = previous + fresh
            chunks.pop()

    return [_render(chunk) for chunk in chunks]


def _render(chunk: list[tuple]) -> dict[str, Any]:
    parts: list[str] = []
    last_paragraph = None
    for _, text, paragraph_index, _size in chunk:
        if parts:
            parts.append(" " if paragraph_index == last_paragraph else PARAGRAPH_SEPARATOR)
        parts.append(text)
        last_paragraph = paragraph_index
    return {
        "text": "".join(parts),
        "words": sum(u[3] for u in chunk),
        "paragraph_start": min(u[2] for u in chunk),
        "paragraph_end": max(u[2] for u in chunk),
    }


def _int_arg(args: dict[str, Any], name: str, default: int, *, minimum: int) -> int:
    value = args.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ToolInputError(
            code="invalid_argument_value",
            message=f"{name} must be an integer >= {minimum}, got {value!r}.",
            tool_name=TOOL_NAME,
            details={"argument": name, "value": value},
        )
    return value


def _header(record: dict[str, Any]) -> str:
    return " — ".join(
        str(record[field]).strip() for field in HEADER_FIELDS if record.get(field) not in (None, "")
    )


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": "Handle of the corpus to chunk, e.g. 'corpus_1'. Omit to use the most recent.",
        },
        "corpus": {
            "type": "string",
            "description": "Name of a registered corpus. Takes precedence over `input`.",
        },
        "name": {
            "type": "string",
            "description": "Name for the chunk corpus (default: '<source>_paragraph_chunks').",
        },
        "target_words": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_TARGET_WORDS,
            "description": "Pack paragraphs until the next one would exceed this many words.",
        },
        "max_words": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_MAX_WORDS,
            "description": "Hard upper bound; longer paragraphs are split at sentences.",
        },
        "min_words": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_MIN_WORDS,
            "description": "A trailing chunk smaller than this joins its predecessor when it fits.",
        },
        "overlap_paragraphs": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Repeat this many trailing paragraphs at the start of the next chunk.",
        },
        "context_header": {
            "type": "boolean",
            "default": False,
            "description": "Prepend the article's title and date to every chunk's text.",
        },
    },
}


class ChunkArticlesByParagraphTool:
    """Split each record of a corpus into paragraph-packed chunks."""

    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE
    tool_name: ClassVar[str] = TOOL_NAME

    schema = ToolSchema(
        name=TOOL_NAME,
        description=(
            "Split the articles of a corpus into chunks of whole paragraphs of about "
            "target_words words, producing a new corpus that retrieve_passages can search. "
            "Each chunk keeps its article id, paragraph span, and the article's metadata."
        ),
        input_schema=_INPUT_SCHEMA,
        render_type="json",
    )

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        target = _int_arg(args, "target_words", DEFAULT_TARGET_WORDS, minimum=1)
        maximum = _int_arg(args, "max_words", DEFAULT_MAX_WORDS, minimum=1)
        minimum = _int_arg(args, "min_words", DEFAULT_MIN_WORDS, minimum=1)
        overlap = _int_arg(args, "overlap_paragraphs", 0, minimum=0)
        if not (minimum <= target <= maximum):
            raise ToolInputError(
                code="invalid_argument_value",
                message=(
                    "Sizes must satisfy min_words <= target_words <= max_words, "
                    f"got {minimum}, {target}, {maximum}."
                ),
                tool_name=TOOL_NAME,
                details={"min_words": minimum, "target_words": target, "max_words": maximum},
            )
        with_header = args.get("context_header") is True

        source, source_ref = bind_corpus(args, context, tool_name=TOOL_NAME)
        # A handle belongs in recordset lineage; a result object's run id does not.
        source_handle = None if isinstance(args.get("input"), dict) else source_ref
        name = str(args.get("name") or "").strip() or f"{source.name}_paragraph_chunks"
        if name == source.name:
            raise ToolInputError(
                code="invalid_argument_value",
                message="The chunk corpus needs a different name from its source corpus.",
                tool_name=TOOL_NAME,
                details={"argument": "name", "value": name},
            )

        rows: list[dict[str, Any]] = []
        chunked_articles = empty_articles = 0
        for record in source.records:
            pieces = chunk_paragraphs(
                record.get(source.text_field),
                target_size=target,
                max_size=maximum,
                min_size=minimum,
                overlap=overlap,
            )
            if not pieces:
                empty_articles += 1
                continue
            chunked_articles += 1
            source_id = str(record.get(source.id_field))
            metadata = {
                k: v for k, v in record.items() if k not in (source.text_field, source.id_field)
            }
            header = _header(record) if with_header else ""
            for index, piece in enumerate(pieces):
                rows.append(
                    {
                        **metadata,
                        "id": f"{source_id}#{index}",
                        "source_id": source_id,
                        "chunk_index": index,
                        "chunk_count": len(pieces),
                        "paragraph_start": piece["paragraph_start"],
                        "paragraph_end": piece["paragraph_end"],
                        "words": piece["words"],
                        "text": f"{header}{PARAGRAPH_SEPARATOR}{piece['text']}"
                        if header
                        else piece["text"],
                    }
                )

        if not rows:
            raise ToolInputError(
                code="nothing_to_chunk",
                message=f"No record in corpus {source.name!r} has text in {source.text_field!r}.",
                tool_name=TOOL_NAME,
                details={"corpus": source.name, "text_field": source.text_field},
            )

        chunks = register_corpus(
            LocalCorpus.from_records(
                rows,
                name=name,
                id_field="id",
                text_field="text",
                source=f"{TOOL_NAME}({source.name})",
                parse_lists=False,
            )
        )
        sizes = [row["words"] for row in rows]
        settings = {
            "target_words": target,
            "max_words": maximum,
            "min_words": minimum,
            "overlap_paragraphs": overlap,
            "context_header": with_header,
        }
        return {
            "corpus": chunks.name,
            "source_corpus": source.name,
            "input_handle": source_handle,
            "handle": self._remember(context, chunks, source, source_handle, settings),
            "articles": chunked_articles,
            "empty_articles": empty_articles,
            "chunks": len(rows),
            "words": {
                "min": min(sizes),
                "median": statistics.median(sizes),
                "max": max(sizes),
            },
            "settings": settings,
            "provenance": make_provenance(
                TOOL_NAME,
                settings={"source_corpus": source.name, "corpus": chunks.name, **settings},
                derived_from=[source_ref],
            ),
        }

    def _remember(self, context, chunks, source, source_handle, settings) -> str | None:
        """Chunks by reference to their registered corpus, with lineage to the source."""
        if context.recordsets is None:
            return None
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=STAGE,
            produced_by=TOOL_NAME,
            args={"corpus": chunks.name, "source_corpus": source.name, **settings},
            ref=local_ref(corpus=chunks.name, ids=chunks.ids),
            count=len(chunks),
            summary=f"{len(chunks)} paragraph chunks from {source.name}",
            derived_from=[source_handle] if source_handle else [],
        )
        return record.handle

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str:
        data = self.run(args, context=context)
        return json.dumps({"render": {"render_type": "json", **data}, **data})
