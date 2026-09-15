"""extract_test_questions — draft a test set of questions from a corpus.

Evaluating retrieval and answers needs questions whose answers are known.
Writing them by hand is the gold standard, and slow; this tool drafts them. It
samples records — articles or chunks — shows each to a language model, and asks
for questions the record answers, each with a short reference answer and a
question type:

    test_set = extract_test_questions(input=chunks, sample_size=120, questions_per_record=2)

Every question keeps where it came from, so one test set serves both halves of
an evaluation:

* **retrieval** — ``source_ids`` names the article the question came from, and
  ``passage_id`` the chunk, when the record was one; that article should be
  retrieved (``calculate_retrieval_metrics``);
* **answers** — ``reference_answer`` says what a correct answer contains
  (``score_rag_answer``), and ``type`` lets results be compared per category.

**Question types.** By default the model is asked for a mix of factual,
temporal, relational, analytical and comparative questions, and labels each.
Restrict or extend the mix with ``question_types`` — or, in a subclass, extend
``question_types`` with your own definitions.

**Review before you trust it.** Generated questions tend to be easier than
real ones — they lean on the source's own words, which favours keyword search —
and other articles may answer a question as well, so a retrieved article that
is not the source is not necessarily wrong: precision measured against the
source alone underestimates. Read the questions, fix or drop weak ones, and add
questions of your own.

**Model.** Draft with a different model than the one that answers, so the
answering model is not tested on questions shaped by its own habits: configure
an ``evaluation`` role (``configure_llm(models={"evaluation": ...})``). Without
one, the default model is used.

**Variants.** Subclass and override ``sample`` (which records — e.g. stratified
by month, entity or subtopic), ``system_prompt`` / ``user_prompt``, ``generate``
or ``clean``.
"""

from __future__ import annotations

import random
import re
from typing import Any, ClassVar

from anatoolbox.base import ToolContext
from anatoolbox.corpus import LocalCorpus, bind_corpus
from anatoolbox.extract.extract.base import PREFIX, STAGE
from anatoolbox.llm_client import call_llm_json, model_for
from anatoolbox.tool import BaseTool

TOOL_NAME = "extract_test_questions"
EVALUATION_ROLE = "evaluation"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_QUESTIONS_PER_RECORD = 1
MAX_QUESTIONS_PER_RECORD = 5
DEFAULT_MIN_CHARS = 500
DEFAULT_MAX_CHARS = 4000
#: Record fields copied onto each question, for reading and filtering the test set.
RECORD_FIELDS = ("title", "date", "url")

QUESTION_TYPES: dict[str, str] = {
    "factual": "asks for a specific fact: who, what, which, how many",
    "temporal": "asks when something happened, or what happened in a period, and needs a date to answer",
    "relational": "asks how two entities are connected: who develops, funds, acquires, partners with or uses what",
    "analytical": "asks why something happened or what it implies, as far as the text explains it",
    "comparative": "asks how two or more things differ or compare, as far as the text says",
}

#: A question that points at "the article" cannot be asked of a whole collection.
_POINTS_AT_THE_SOURCE = re.compile(
    r"\b(?:this|the|that)\s+(?:article|text|passage|document|report|piece|story|excerpt)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You write test questions for evaluating a search and question-answering system over a collection of news articles.

You are shown one text from the collection. Write up to {n} question(s) that this text answers.

Question types — use a mix, as far as the text supports them, and label each question with its type:
{types}

Rules:
- Each question must make sense on its own to someone searching the whole collection: name the companies, products, people or events it is about. Never refer to "the article" or "this text".
- Every question must be answerable from this text alone. Do not ask what the text does not say.
- Phrase questions the way a person would ask them; do not copy sentences from the text.
- answer: a short reference answer, one or two sentences, using only facts stated in the text. For temporal questions, include the date."""

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input": {
            "type": "string",
            "description": "The corpus to draw questions from: a handle such as 'corpus_1'.",
        },
        "corpus": {"type": "string", "description": "Name of a registered corpus."},
        "sample_size": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_SAMPLE_SIZE,
            "description": "How many records to write questions for.",
        },
        "seed": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Random seed for the sample, so the same records are drawn again.",
        },
        "questions_per_record": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_QUESTIONS_PER_RECORD,
            "default": DEFAULT_QUESTIONS_PER_RECORD,
        },
        "question_types": {
            "type": "array",
            "items": {"type": "string", "enum": list(QUESTION_TYPES)},
            "description": "Which question types to ask for (default: all).",
        },
        "min_chars": {
            "type": "integer",
            "minimum": 0,
            "default": DEFAULT_MIN_CHARS,
            "description": "Only sample records with at least this many characters of text.",
        },
        "max_chars": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_MAX_CHARS,
            "description": "Show the model at most this many characters of each record.",
        },
        "model": {
            "type": "string",
            "description": "Model name; defaults to the evaluation role, else the default model.",
        },
    },
}


class ExtractTestQuestionsTool(BaseTool):
    """Draft questions with reference answers and question types from sampled records.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``sample(corpus, settings)`` — which records to write questions for.
    * ``question_types`` (class attribute) — type name -> definition; extend it.
    * ``system_prompt(settings)`` / ``user_prompt(record, text, settings)`` — the instructions.
    * ``generate(system, user, settings, context)`` — the raw reply (default: one JSON LLM call).
    * ``clean(reply, record, settings)`` — turn the reply into question dicts.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Draft a test set: sample records from a corpus and have a language model write "
        "questions each record answers — factual, temporal, relational, analytical or "
        "comparative — with reference answers and the source article and passage. Review the "
        "questions before evaluating with them."
    )
    input_schema = _INPUT_SCHEMA
    render_type = "table"
    role: ClassVar[str] = EVALUATION_ROLE
    question_types: ClassVar[dict[str, str]] = QUESTION_TYPES

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        """Checked arguments. Recorded in provenance, including the model actually used."""
        per_record = self.int_arg(args, "questions_per_record", DEFAULT_QUESTIONS_PER_RECORD)
        if per_record > MAX_QUESTIONS_PER_RECORD:
            raise self.input_error(
                f"questions_per_record must be at most {MAX_QUESTIONS_PER_RECORD}, got {per_record}.",
                argument="questions_per_record",
                value=per_record,
            )
        types = args.get("question_types")
        if types is None:
            types = list(self.question_types)
        elif (
            not isinstance(types, list)
            or not types
            or not all(isinstance(t, str) and t in self.question_types for t in types)
        ):
            raise self.input_error(
                f"question_types must be a non-empty list drawn from {list(self.question_types)}, "
                f"got {types!r}.",
                argument="question_types",
                value=types,
            )
        return {
            "sample_size": self.int_arg(args, "sample_size", DEFAULT_SAMPLE_SIZE),
            "seed": self.int_arg(args, "seed", 0, minimum=0),
            "questions_per_record": per_record,
            "question_types": list(dict.fromkeys(types)),
            "min_chars": self.int_arg(args, "min_chars", DEFAULT_MIN_CHARS, minimum=0),
            "max_chars": self.int_arg(args, "max_chars", DEFAULT_MAX_CHARS),
            "model": self.text_arg(args, "model") or model_for(self.role),
        }

    def sample(self, corpus: LocalCorpus, settings: dict[str, Any]) -> list[dict[str, Any]]:
        """Records to write questions for. Default: a seeded random sample of records
        with at least ``min_chars`` characters of text."""
        eligible = [r for r in corpus.records if len(corpus.text_of(r)) >= settings["min_chars"]]
        size = min(settings["sample_size"], len(eligible))
        return random.Random(settings["seed"]).sample(eligible, size)

    def system_prompt(self, settings: dict[str, Any]) -> str:
        """The system prompt: the task, the question types and the rules."""
        return SYSTEM_PROMPT.format(
            n=settings["questions_per_record"],
            types="\n".join(
                f"- {name}: {self.question_types[name]}" for name in settings["question_types"]
            ),
        )

    def user_prompt(self, record: dict[str, Any], text: str, settings: dict[str, Any]) -> str:
        """The user message: one record's title, date and text."""
        lines = []
        if record.get("title"):
            lines.append(f"Title: {record['title']}")
        if record.get("date"):
            lines.append(f"Published: {record['date']}")
        body = text[: settings["max_chars"]]
        return "\n".join(lines) + ("\n\n" if lines else "") + body

    def response_schema(self, settings: dict[str, Any]) -> dict[str, Any]:
        """The JSON schema the reply must follow, strictly where the endpoint supports it."""
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["questions"],
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["question", "answer", "type"],
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "type": {"type": "string", "enum": settings["question_types"]},
                        },
                    },
                }
            },
        }

    def generate(
        self, system: str, user: str, settings: dict[str, Any], context: ToolContext
    ) -> Any:
        """The raw reply: ``{"questions": [{"question": ..., "answer": ..., "type": ...}]}``."""
        return call_llm_json(
            system,
            user,
            model=settings["model"],
            temperature=0.0,
            response_schema=self.response_schema(settings),
            project=context.project,
        )

    def clean(
        self, reply: Any, record: dict[str, Any], settings: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Usable questions: non-empty, not repeated, not pointing at "the article"; capped.

        A type outside ``question_types`` becomes None rather than dropping the question.
        """
        items = reply.get("questions") if isinstance(reply, dict) else None
        questions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items or []:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            answer = str(item.get("answer") or "").strip()
            if not question or not answer or _POINTS_AT_THE_SOURCE.search(question):
                continue
            if question.casefold() in seen:
                continue
            seen.add(question.casefold())
            kind = str(item.get("type") or "").strip().casefold()
            questions.append(
                {
                    "question": question,
                    "reference_answer": answer,
                    "type": kind if kind in settings["question_types"] else None,
                }
            )
        return questions[: settings["questions_per_record"]]

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        settings = self.settings(args)
        corpus, source_ref = bind_corpus(args, context, tool_name=self.tool_name)
        records = self.sample(corpus, settings)
        if not records:
            raise self.input_error(
                f"No record in corpus {corpus.name!r} has at least {settings['min_chars']} "
                "characters of text; lower min_chars.",
                code="nothing_to_sample",
                corpus=corpus.name,
            )

        system = self.system_prompt(settings)
        questions: list[dict[str, Any]] = []
        skipped: list[str] = []
        for record in records:
            record_id = str(record.get(corpus.id_field))
            # A chunk names its article in source_id; an article is its own source.
            article_id = str(record.get("source_id") or record_id)
            user = self.user_prompt(record, corpus.text_of(record), settings)
            drafted = self.clean(self.generate(system, user, settings, context), record, settings)
            if not drafted:
                skipped.append(record_id)
                continue
            for item in drafted:
                questions.append(
                    {
                        "id": f"q{len(questions) + 1}",
                        "question": item["question"],
                        "reference_answer": item.get("reference_answer"),
                        "type": item.get("type"),
                        **{
                            k: v
                            for k, v in item.items()
                            if k not in ("id", "question", "reference_answer", "type")
                        },
                        "source_ids": [article_id],
                        "passage_id": record_id if article_id != record_id else None,
                        **{f: record.get(f) for f in RECORD_FIELDS if record.get(f) is not None},
                    }
                )

        return {
            "corpus": corpus.name,
            "model": settings["model"],
            "records_sampled": len(records),
            "questions": questions,
            "types": {
                kind: sum(1 for q in questions if q["type"] == kind)
                for kind in settings["question_types"]
            },
            "skipped_records": skipped,
            "input_handle": None if isinstance(args.get("input"), dict) else source_ref,
            "provenance": self.provenance(
                {**settings, "corpus": corpus.name}, derived_from=[source_ref]
            ),
        }

    def render(self, data: dict[str, Any]) -> dict[str, Any]:
        columns = ["id", "type", "question", "reference_answer", "source_ids"]
        return {
            "render_type": "table",
            "columns": columns,
            "rows": [{k: q.get(k) for k in columns} for q in data["questions"]],
            "caption": f"{len(data['questions'])} draft questions from {data['corpus']}",
        }
