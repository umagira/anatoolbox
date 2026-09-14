"""synthesize_answer — answer a question from retrieved passages, with citations.

The generation half of RAG. It takes the passages a retrieval (and optionally
a rerank) produced, curates them into a prompt, and asks a language model for
an answer in which claims point back at numbered sources:

    retrieve_passages(query="...", size=30)
    rerank_passages(keep=8)
    synthesize_answer(question="...")        # binds the reranked passages

What happens between retrieval and the model matters as much as either:

* **Numbered sources, checked citations.** Passages are labelled [S1]…[Sn]
  with their title, publication date and url, and the answer cites labels.
  The tool checks them: labels that were never provided come back as
  ``unknown_citations``, sources the answer never used as
  ``uncited_sources``. Neither proves an answer faithful, but both are cheap,
  objective signals to evaluate against — and small models do invent labels.
  ``citation_coverage`` counts how many of the answer's sentences carry an
  inline citation; a model that lists every label at the end scores 0.
* **Two answering modes.** ``grounded`` restricts the answer to what the
  sources say and asks the model to say when they are not enough.
  ``blended`` lets it add background knowledge, marked as background and
  never cited — for exploratory questions where the sources alone are thin.
* **Context curation.** Duplicate passages are dropped before prompting, so
  the model is not shown the same evidence twice. Models attend least to the
  middle of a long context (Liu et al., 2023, "Lost in the Middle"), so by
  default the most relevant passages go at the start and the end. After
  chunking, several chunks of one article can crowd the context;
  ``max_per_source`` caps them.
* **Dates in view.** Every source shows its publication date, and the
  instructions ask the model to use dates when the question concerns time.

The model comes from the ``strong`` role of ``anatoolbox.llm_client``, falling
back to the default model; the model actually used is recorded with the answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from anatoolbox.base import ToolContext, ToolSchema
from anatoolbox.corpus import get_corpus
from anatoolbox.enrich.synthesize.base import PREFIX, STAGE
from anatoolbox.errors import ToolInputError
from anatoolbox.llm_client import call_llm_text, model_for
from anatoolbox.memory import value_ref
from anatoolbox.render import MARKDOWN_RENDER_TYPE

TOOL_NAME = "synthesize_answer"
OBJECT_TYPE = "answer"
PASSAGES_OBJECT_TYPE = "passages"
MODES = ("grounded", "blended")
ORDERS = ("edges", "relevance")
DEFAULT_MODE = "grounded"
DEFAULT_ORDER = "edges"
DEFAULT_MAX_PASSAGES = 8
DEFAULT_MAX_CHARS = 1500
#: Source metadata shown next to each label, in this order.
SOURCE_FIELDS = ("title", "date", "url")

_CITATION_GROUP = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]")
_LABEL = re.compile(r"S\d+")
#: A sentence ends at . ! or ? — possibly followed by a closing quote or bracket.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"”'’)\]])\s+")
_WORD = re.compile(r"[^\W_]")
#: A line that begins with a label is a reference-list entry, not a statement.
_REFERENCE_ENTRY = re.compile(r"^(?:[-*]\s*)?\[\s*S\d+")
_SOURCE_LIST_HEADINGS = {"sources", "source", "references", "bibliography"}

SYSTEM_PROMPT = """You answer questions using numbered sources from a document collection.

Rules:
{grounding}
- Cite with the source labels in square brackets right after the sentence they support, for example [S1] or [S2][S4]. Do not collect citations at the end of the answer.
- Use only the labels you are given. Never invent sources, URLs, names, figures, or dates.
- When the question concerns time — what is new, what changed, when something happened — use the sources' publication dates and mention them.
- Where sources disagree, say so and cite each side.
- Write concise markdown. Do not end with a list of sources; the application adds one."""

GROUNDING = {
    "grounded": (
        "- Base every statement on the sources. If they do not contain enough information "
        "to answer, say so plainly and answer only what they support."
    ),
    "blended": (
        "- Base statements about the specifics of the question on the sources. You may add "
        "general background knowledge where it helps, but mark it clearly as background and "
        "never attach a source label to it."
    ),
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def curate(
    passages: list[dict[str, Any]],
    *,
    max_passages: int = DEFAULT_MAX_PASSAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
    order: str = DEFAULT_ORDER,
    max_per_source: int | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Deduplicate, cap, truncate, label and order passages for a prompt.

    ``passages`` arrive in relevance order, each with a ``text``. Labels follow
    relevance (S1 is the most relevant); ``position`` is the place in the
    prompt. ``max_per_source`` keeps at most that many passages per article
    (``source_id``, else ``id``) so one source cannot crowd out the rest.
    Returns the curated sources in prompt order, the number of duplicates
    dropped, and the number dropped by the per-source limit.
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    duplicates = 0
    for passage in passages:
        key = _normalized(str(passage.get("text") or ""))[:300]
        if not key:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        kept.append(passage)

    over_limit = 0
    if max_per_source is not None:
        per_source: dict[str, int] = {}
        diverse = []
        for passage in kept:
            key = str(passage.get("source_id") or passage.get("id"))
            if per_source.get(key, 0) >= max_per_source:
                over_limit += 1
                continue
            per_source[key] = per_source.get(key, 0) + 1
            diverse.append(passage)
        kept = diverse

    labelled = []
    for rank, passage in enumerate(kept[:max_passages], start=1):
        text = str(passage["text"])
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + " …"
        labelled.append({**passage, "label": f"S{rank}", "rank": rank, "text": text})

    if order == "edges":
        front, back = [], []
        for index, source in enumerate(labelled):
            (front if index % 2 == 0 else back).append(source)
        labelled = front + back[::-1]
    for position, source in enumerate(labelled, start=1):
        source["position"] = position
    return labelled, duplicates, over_limit


def build_user_prompt(
    question: str, sources: list[dict[str, Any]], *, instructions: str | None = None
) -> str:
    blocks = []
    for source in sources:
        meta = " · ".join(str(source[f]) for f in SOURCE_FIELDS if source.get(f))
        header = f"[{source['label']}] {meta}" if meta else f"[{source['label']}]"
        blocks.append(f"{header}\n{source['text']}")
    parts = [f"Question: {question}", "Sources:\n\n" + "\n\n".join(blocks)]
    if instructions:
        parts.append(f"Additional instructions: {instructions}")
    # Keep this plain. A format example here ('... like this: "This is what the
    # source says [S1]."') was tried, and Qwen3-0.6B copied the example sentence
    # verbatim into its answers, once per source.
    parts.append("Answer the question, citing sources by label.")
    return "\n\n".join(parts)


def extract_citations(answer: str) -> list[str]:
    """Labels cited in an answer, in order of first appearance. Accepts [S1][S2] and [S1, S2]."""
    labels: list[str] = []
    for group in _CITATION_GROUP.finditer(answer):
        for label in _LABEL.findall(group.group(1)):
            if label not in labels:
                labels.append(label)
    return labels


def citation_coverage(answer: str) -> dict[str, Any]:
    """How many of an answer's statements carry an inline citation.

    A statement is a sentence with words in it. A citation-only piece right
    after a sentence on the same line ("... restricted. [S2]") counts for that
    sentence; a line made only of citations — small models like to list every
    label at the end — supports no particular sentence and counts for nothing.
    Headings are not statements. A source list the model writes itself — a
    "Sources:" heading, or lines that begin with a label — is not counted:
    it attributes no sentence. This is a count, not a verdict: a cited
    sentence can still misstate its source.
    """
    cited_flags: list[bool] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.strip("*#_ ").rstrip(":").strip().casefold() in _SOURCE_LIST_HEADINGS:
            break
        if stripped.startswith("#") or _REFERENCE_ENTRY.match(stripped):
            continue
        previous = None
        for piece in _SENTENCE_END.split(line):
            has_citation = bool(_CITATION_GROUP.search(piece))
            if _WORD.search(_CITATION_GROUP.sub("", piece)):
                cited_flags.append(has_citation)
                previous = len(cited_flags) - 1
            elif has_citation and previous is not None:
                cited_flags[previous] = True
    statements = len(cited_flags)
    with_citations = sum(cited_flags)
    return {
        "statements": statements,
        "statements_with_citations": with_citations,
        "citation_coverage": round(with_citations / statements, 3) if statements else None,
    }


def link_citations(answer: str, sources_by_label: dict[str, dict[str, Any]]) -> str:
    """Turn each cited label into a markdown link to its source url, when it has one."""

    def replace(match: re.Match) -> str:
        links = []
        for label in _LABEL.findall(match.group(1)):
            url = (sources_by_label.get(label) or {}).get("url")
            links.append(f"[{label}]({url})" if url else f"[{label}]")
        return "".join(links)

    return _CITATION_GROUP.sub(replace, answer)


def _positive_int(args: dict[str, Any], name: str, default: int | None) -> int | None:
    value = args.get(name, default)
    if value is None and default is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ToolInputError(
            code="invalid_argument_value",
            message=f"{name} must be a positive integer, got {value!r}.",
            tool_name=TOOL_NAME,
            details={"argument": name, "value": value},
        )
    return value


def _choice(args: dict[str, Any], name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = str(args.get(name) or default).strip()
    if value not in allowed:
        raise ToolInputError(
            code="invalid_argument_value",
            message=f"{name} must be one of {list(allowed)}, got {value!r}.",
            tool_name=TOOL_NAME,
            details={"argument": name, "value": value},
        )
    return value


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "The question to answer."},
        "input": {
            "type": "string",
            "description": "Handle of the passages to answer from, e.g. 'passages_2'. Omit to use the most recent.",
        },
        "mode": {
            "type": "string",
            "enum": list(MODES),
            "default": DEFAULT_MODE,
            "description": (
                "grounded = only what the sources say; blended = may add clearly marked "
                "background knowledge."
            ),
        },
        "order": {
            "type": "string",
            "enum": list(ORDERS),
            "default": DEFAULT_ORDER,
            "description": (
                "edges = most relevant passages at the start and end of the prompt "
                "(against lost-in-the-middle); relevance = in rank order."
            ),
        },
        "max_passages": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_PASSAGES},
        "max_chars_per_passage": {"type": "integer", "minimum": 1, "default": DEFAULT_MAX_CHARS},
        "max_per_source": {
            "type": "integer",
            "minimum": 1,
            "description": "At most this many passages from one article (by source_id).",
        },
        "instructions": {
            "type": "string",
            "description": "Optional extra instructions, e.g. audience, format or length.",
        },
        "model": {
            "type": "string",
            "description": "Model name; defaults to the configured strong role.",
        },
        "max_tokens": {"type": "integer", "minimum": 1},
        "passages": {
            "type": "array",
            "items": {"type": "object"},
            "description": (
                "Optional explicit passages in relevance order, each with 'id' and 'text' "
                "(and optionally title, date, url), for pipelines."
            ),
        },
    },
    "required": ["question"],
}


class SynthesizeAnswerTool:
    """Answer a question from retrieved passages, citing numbered sources."""

    prefix: ClassVar[str] = PREFIX
    stage: ClassVar[str] = STAGE
    tool_name: ClassVar[str] = TOOL_NAME
    role: ClassVar[str] = "strong"

    schema = ToolSchema(
        name=TOOL_NAME,
        description=(
            "Write an answer to a question from retrieved passages, citing numbered sources "
            "[S1]…[Sn]. Binds the most recent passages (retrieved or reranked) unless given a "
            "handle, and reports citations to unknown labels and sources left uncited."
        ),
        input_schema=_INPUT_SCHEMA,
        render_type=MARKDOWN_RENDER_TYPE,
    )

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        question = str(args.get("question") or "").strip()
        if not question:
            raise ToolInputError(
                code="missing_required_arguments",
                message="Argument 'question' is required.",
                tool_name=TOOL_NAME,
                details={"missing": ["question"]},
            )
        mode = _choice(args, "mode", DEFAULT_MODE, MODES)
        order = _choice(args, "order", DEFAULT_ORDER, ORDERS)
        max_passages = _positive_int(args, "max_passages", DEFAULT_MAX_PASSAGES)
        max_chars = _positive_int(args, "max_chars_per_passage", DEFAULT_MAX_CHARS)
        max_tokens = _positive_int(args, "max_tokens", None)
        max_per_source = _positive_int(args, "max_per_source", None)
        instructions = str(args.get("instructions") or "").strip() or None

        passages, input_handle = self._passages(args, context=context)
        sources, duplicates, over_limit = curate(
            passages,
            max_passages=max_passages,
            max_chars=max_chars,
            order=order,
            max_per_source=max_per_source,
        )
        if not sources:
            raise ToolInputError(
                code="no_candidates",
                message="There are no passages with text to answer from.",
                tool_name=TOOL_NAME,
                details={"input_handle": input_handle},
            )

        model = str(args.get("model") or "").strip() or model_for(self.role)
        answer = call_llm_text(
            SYSTEM_PROMPT.format(grounding=GROUNDING[mode]),
            build_user_prompt(question, sources, instructions=instructions),
            model=model,
            temperature=0.0,
            project=context.project,
            max_tokens=max_tokens,
        ).strip()

        by_label = {source["label"]: source for source in sources}
        cited = extract_citations(answer)
        by_rank = sorted(sources, key=lambda source: source["rank"])
        result = {
            "question": question,
            "max_per_source": max_per_source,
            "mode": mode,
            "order": order,
            "model": model,
            "answer": answer,
            "answer_markdown": link_citations(answer, by_label),
            "sources": [
                {
                    "label": s["label"],
                    "rank": s["rank"],
                    "position": s["position"],
                    "id": s["id"],
                    **{field: s.get(field) for field in ("source_id", "title", "date", "url")},
                }
                for s in by_rank
            ],
            "cited": [label for label in cited if label in by_label],
            "unknown_citations": [label for label in cited if label not in by_label],
            "uncited_sources": [s["label"] for s in by_rank if s["label"] not in cited],
            "dropped_duplicates": duplicates,
            "dropped_over_source_limit": over_limit,
            **citation_coverage(answer),
            "input_handle": input_handle,
        }
        result["handle"] = self._remember(context, result, max_passages=max_passages)
        return result

    def _passages(self, args: dict[str, Any], *, context: ToolContext):
        """Explicit ``passages`` win; otherwise bind a stored passages recordset (full texts)."""
        explicit = args.get("passages")
        if explicit is not None:
            if not isinstance(explicit, list) or not all(
                isinstance(p, dict) and "id" in p and "text" in p for p in explicit
            ):
                raise ToolInputError(
                    code="invalid_argument_value",
                    message="passages must be a list of objects with 'id' and 'text'.",
                    tool_name=TOOL_NAME,
                    details={"argument": "passages"},
                )
            return [{**p, "id": str(p["id"]), "text": str(p["text"])} for p in explicit], None

        if context.recordsets is None:
            raise ToolInputError(
                code="missing_required_arguments",
                message="No passages: pass passages=[...], or run retrieve_passages first.",
                tool_name=TOOL_NAME,
                details={"missing": ["passages"]},
            )
        requested = args.get("input")
        record = context.recordsets.bind(
            object_type=PASSAGES_OBJECT_TYPE,
            tool_name=TOOL_NAME,
            requested=requested.strip()
            if isinstance(requested, str) and requested.strip()
            else None,
        )
        corpus_name = str(record.ref.get("corpus") or "")
        try:
            corpus = get_corpus(corpus_name)
        except KeyError as exc:
            raise ToolInputError(
                code="unknown_corpus",
                message=str(exc.args[0]) if exc.args else f"No corpus named {corpus_name!r}.",
                tool_name=TOOL_NAME,
                details={"corpus": corpus_name, "input_handle": record.handle},
            ) from exc
        passages = []
        for row in corpus.get(record.ref.get("ids") or []):
            passages.append(
                {
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in (corpus.text_field, corpus.id_field)
                    },
                    "id": str(row.get(corpus.id_field)),
                    "text": corpus.text_of(row),
                }
            )
        return passages, record.handle

    def _remember(
        self, context: ToolContext, result: dict[str, Any], *, max_passages: int
    ) -> str | None:
        """The answer by value: it exists nowhere else. Lineage points at its passages."""
        if context.recordsets is None:
            return None
        record = context.recordsets.remember(
            object_type=OBJECT_TYPE,
            stage=STAGE,
            produced_by=TOOL_NAME,
            args={
                "question": result["question"],
                "mode": result["mode"],
                "order": result["order"],
                "model": result["model"],
                "max_passages": max_passages,
                "max_per_source": result["max_per_source"],
            },
            ref=value_ref(
                [
                    {
                        "question": result["question"],
                        "answer": result["answer"],
                        "cited": result["cited"],
                        "citation_coverage": result["citation_coverage"],
                        "sources": [
                            {"label": s["label"], "id": s["id"]} for s in result["sources"]
                        ],
                    }
                ]
            ),
            count=1,
            summary=(
                f"answer to {result['question']!r} citing {len(result['cited'])} of "
                f"{len(result['sources'])} sources"
            ),
            derived_from=[result["input_handle"]] if result["input_handle"] else [],
        )
        return record.handle

    def execute(self, args: dict[str, Any], *, context: ToolContext) -> str:
        data = self.run(args, context=context)
        lines = []
        for source in data["sources"]:
            meta = " · ".join(str(source[f]) for f in ("title", "date") if source.get(f))
            url = f" — {source['url']}" if source.get("url") else ""
            lines.append(f"- **[{source['label']}]** {meta}{url}")
        content = data["answer_markdown"] + "\n\n**Sources**\n\n" + "\n".join(lines)
        return json.dumps(
            {"render": {"render_type": MARKDOWN_RENDER_TYPE, "content": content}, **data}
        )
