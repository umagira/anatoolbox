"""score_rag_answer — judge a generated answer with a language model.

The citation checks of ``synthesize_answer`` are objective but shallow: a cited
sentence can still misstate its source. A judge model reads the question, the
evidence the answer was written from — sources and any graph facts — and the
answer, and scores the answer from 1 to 5 on each criterion, with a
one-sentence rationale:

* ``faithfulness`` — is every statement supported by the evidence? (also
  called groundedness)
* ``relevance`` — does the answer address the question?
* ``context_grounding`` — does it use the relevant evidence, passages and graph
  facts, and cite it where it does?
* ``temporal_grounding`` — are dates and chronology right? Scored null when the
  question does not concern time.
* ``correctness`` — does it agree with a reference answer? Only when one is given.

    score_rag_answer(answer=answer_result, reference_answer=q["reference_answer"])

It also lists ``unsupported_claims`` — statements the judge found no support
for — which turn a low faithfulness score into something to look at. A score
of None means the criterion did not apply, or the judge gave no usable score;
the rationale says which.

``review_flags`` lists signs that the judgment itself needs a human look:
a top faithfulness score next to listed unsupported claims, a rationale that
only repeats the criterion's description, a criterion left without score or
rationale. Small judge models produce all three.

**A judge is a model too.** Use a different, preferably stronger, model than the
one that answered (configure the ``evaluation`` role; without it, the default
model is used), and check a sample of its scores by hand before comparing
systems on them. Compare systems only on scores from the same judge model.

**Variants.** Subclass and extend ``criteria`` with your own, or override
``system_prompt`` / ``user_prompt``, ``judge`` or ``clean``.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from anatoolbox.analyze.score.base import PREFIX, STAGE
from anatoolbox.base import ToolContext
from anatoolbox.corpus import passages_with_text
from anatoolbox.graph import describe_fact
from anatoolbox.llm_client import call_llm_json, model_for
from anatoolbox.provenance import run_id_of
from anatoolbox.tool import BaseTool

TOOL_NAME = "score_rag_answer"
EVALUATION_ROLE = "evaluation"
MIN_SCORE, MAX_SCORE = 1, 5
DEFAULT_MAX_CHARS = 1500

CRITERIA: dict[str, str] = {
    "faithfulness": (
        "Every statement in the answer is supported by the sources and graph facts it was "
        "given. 5: all statements are supported. 3: some statements are unsupported. 1: most "
        "statements are unsupported or contradict the evidence."
    ),
    "relevance": (
        "The answer addresses the question directly and completely. 5: it fully answers the "
        "question. 3: it answers part of it or drifts off topic. 1: it does not answer it."
    ),
    "context_grounding": (
        "The answer makes appropriate use of the evidence: it draws on the sources and graph "
        "facts that bear on the question and cites them where it uses them. 5: it uses the "
        "relevant evidence and cites it. 3: it misses relevant evidence or cites loosely. 1: it "
        "ignores the evidence or cites it wrongly."
    ),
    "temporal_grounding": (
        "Dates and chronology in the answer are correct according to the evidence and its "
        "publication dates, and the answer says when things happened where that matters. 5: "
        "correct and clear. 3: vague or partly wrong. 1: wrong dates or order of events. Score "
        "null if the question does not concern time."
    ),
    "correctness": (
        "The answer agrees with the reference answer on the facts that matter. 5: it agrees on "
        "all key facts. 3: it misses or blurs some key facts. 1: it contradicts the reference "
        "answer or misses it entirely."
    ),
}
DEFAULT_CRITERIA = ("faithfulness", "relevance", "context_grounding", "temporal_grounding")
#: Criteria that need a reference answer.
NEEDS_REFERENCE = ("correctness",)

SYSTEM_PROMPT = """You evaluate answers written by a question-answering system from numbered evidence: sources [S1]… and, where given, graph facts [G1]….

For each criterion below, give an integer score from {low} to {high} and a one-sentence rationale. Judge only by the criterion's description; do not reward length or style. If a criterion does not apply to this question, give the score null and say why in the rationale. Then list every statement in the answer that the evidence does not support in unsupported_claims, or an empty list if there are none.

Criteria:
{criteria}"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scores", "unsupported_claims"],
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["criterion", "score", "rationale"],
                "properties": {
                    "criterion": {"type": "string"},
                    "score": {"type": ["integer", "null"]},
                    "rationale": {"type": "string"},
                },
            },
        },
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
}

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "object",
            "description": "The result of synthesize_answer to evaluate.",
        },
        "reference_answer": {
            "type": "string",
            "description": "What a correct answer says. Enables the correctness criterion.",
        },
        "criteria": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Criteria to score. Default: faithfulness, relevance, context_grounding and "
                "temporal_grounding, plus correctness when a reference answer is given."
            ),
        },
        "corpus": {
            "type": "string",
            "description": "Corpus the answer's sources come from. Default: the one the answer names.",
        },
        "passages": {
            "type": "array",
            "items": {"type": "object"},
            "description": "The answer's passages with 'id' and 'text', when they are not in a corpus.",
        },
        "max_chars_per_source": {
            "type": "integer",
            "minimum": 1,
            "default": DEFAULT_MAX_CHARS,
            "description": "Show the judge at most this many characters of each source.",
        },
        "model": {
            "type": "string",
            "description": "Judge model; defaults to the evaluation role, else the default model.",
        },
    },
    "required": ["answer"],
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().rstrip(".").casefold()


class ScoreRagAnswerTool(BaseTool):
    """Score an answer on faithfulness, relevance, grounding and correctness with a judge model.

    Hooks for a variant (override in a subclass with its own ``tool_name``):

    * ``criteria`` (class attribute) — criterion name -> description; extend it.
    * ``system_prompt(settings)`` / ``user_prompt(question, answer, sources, settings, facts)``.
    * ``judge(system, user, settings, context)`` — the raw reply (default: one JSON LLM call).
    * ``clean(reply, settings)`` — scores, rationales and unsupported claims.
    * ``review(judged, settings)`` — flags that a judgment needs a human look.
    * ``settings(args)`` — read and check arguments; add your own here.
    """

    tool_name = TOOL_NAME
    prefix = PREFIX
    stage = STAGE
    description = (
        "Judge an answer from synthesize_answer with a language model: a 1-5 score and rationale "
        "for faithfulness to its evidence, relevance, context grounding, temporal grounding and "
        "(with a reference answer) correctness, plus the claims the evidence does not support."
    )
    input_schema = _INPUT_SCHEMA
    role: ClassVar[str] = EVALUATION_ROLE
    criteria: ClassVar[dict[str, str]] = CRITERIA

    # --- hooks -------------------------------------------------------------

    def settings(self, args: dict[str, Any]) -> dict[str, Any]:
        reference = self.text_arg(args, "reference_answer") or None
        requested = args.get("criteria")
        if requested is None:
            names = list(DEFAULT_CRITERIA) + (["correctness"] if reference else [])
        elif (
            isinstance(requested, list) and requested and all(isinstance(n, str) for n in requested)
        ):
            names = list(dict.fromkeys(n.strip() for n in requested))
        else:
            raise self.input_error(
                "criteria must be a non-empty list of criterion names.", argument="criteria"
            )
        unknown = [n for n in names if n not in self.criteria]
        if unknown:
            raise self.input_error(
                f"Unknown criteria {unknown}. Choose from {sorted(self.criteria)}.",
                argument="criteria",
                value=unknown,
            )
        missing_reference = [n for n in names if n in NEEDS_REFERENCE and not reference]
        if missing_reference:
            raise self.input_error(
                f"{missing_reference} need a reference_answer.", argument="reference_answer"
            )
        return {
            "criteria": names,
            "reference_answer": reference,
            "max_chars_per_source": self.int_arg(args, "max_chars_per_source", DEFAULT_MAX_CHARS),
            "model": self.text_arg(args, "model") or model_for(self.role),
        }

    def system_prompt(self, settings: dict[str, Any]) -> str:
        return SYSTEM_PROMPT.format(
            low=MIN_SCORE,
            high=MAX_SCORE,
            criteria="\n".join(f"- {name}: {self.criteria[name]}" for name in settings["criteria"]),
        )

    def user_prompt(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        settings: dict[str, Any],
        facts: list[dict[str, Any]] | tuple = (),
    ) -> str:
        limit = settings["max_chars_per_source"]
        blocks = []
        for source in sources:
            meta = " · ".join(str(source[f]) for f in ("title", "date") if source.get(f))
            text = source["text"] if len(source["text"]) <= limit else source["text"][:limit] + " …"
            blocks.append(f"[{source['label']}] {meta}".rstrip() + f"\n{text}")
        parts = [f"Question: {question}", "Sources:\n\n" + "\n\n".join(blocks)]
        if facts:
            parts.append(
                "Graph facts:\n\n"
                + "\n".join(f"[{fact['label']}] {describe_fact(fact)}" for fact in facts)
            )
        if settings["reference_answer"]:
            parts.append(f"Reference answer: {settings['reference_answer']}")
        parts.append(f"Answer to evaluate:\n{answer}")
        return "\n\n".join(parts)

    def judge(self, system: str, user: str, settings: dict[str, Any], context: ToolContext) -> Any:
        """The raw reply: ``{"scores": [{criterion, score, rationale}], "unsupported_claims": [...]}``."""
        return call_llm_json(
            system,
            user,
            model=settings["model"],
            temperature=0.0,
            response_schema=RESPONSE_SCHEMA,
            project=context.project,
        )

    def clean(self, reply: Any, settings: dict[str, Any]) -> dict[str, Any]:
        """Scores clamped to the scale; a criterion that does not apply, or that the
        judge skipped, scores None.

        Besides the requested shape, accepts ``scores`` as a mapping of criterion to a
        score or to ``{score, rationale}`` — small models often answer that way.
        """
        reply = reply if isinstance(reply, dict) else {}
        raw = reply.get("scores")
        entries: dict[str, Any] = {}
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict) and item.get("criterion"):
                    entries[str(item["criterion"]).strip()] = item
        elif isinstance(raw, dict):
            entries = {str(k).strip(): v for k, v in raw.items()}

        scores: dict[str, int | None] = {}
        rationales: dict[str, str | None] = {}
        for name in settings["criteria"]:
            entry = entries.get(name)
            value = entry.get("score") if isinstance(entry, dict) else entry
            try:
                score = int(round(float(value)))
            except (TypeError, ValueError):
                score = None
            scores[name] = None if score is None else max(MIN_SCORE, min(MAX_SCORE, score))
            rationale = entry.get("rationale") if isinstance(entry, dict) else None
            rationales[name] = str(rationale).strip() if rationale else None
        claims = [str(c).strip() for c in reply.get("unsupported_claims") or [] if str(c).strip()]
        return {"scores": scores, "rationales": rationales, "unsupported_claims": claims}

    def review(self, judged: dict[str, Any], settings: dict[str, Any]) -> list[str]:
        """Signs that a judgment needs a human look — objective checks on the judge's own output."""
        scores, rationales = judged["scores"], judged["rationales"]
        flags = []
        for name in settings["criteria"]:
            rationale = _normalized(rationales.get(name) or "")
            if scores.get(name) is None and not rationale:
                flags.append(f"{name}: no score and no rationale")
            if len(rationale) >= 20 and rationale in _normalized(self.criteria.get(name, "")):
                flags.append(
                    f"{name}: the rationale repeats the criterion instead of explaining the score"
                )
        if judged["unsupported_claims"] and scores.get("faithfulness") == MAX_SCORE:
            flags.append("faithfulness: top score, yet unsupported claims are listed")
        return flags

    # --- the fixed part ----------------------------------------------------

    def run(self, args: dict[str, Any], *, context: ToolContext) -> dict[str, Any]:
        answer = args.get("answer")
        if not isinstance(answer, dict) or not isinstance(answer.get("answer"), str):
            raise self.input_error(
                "answer must be the result of synthesize_answer (with 'question', 'answer' and 'sources').",
                code="missing_required_arguments",
                missing=["answer"],
            )
        settings = self.settings(args)
        question = str(answer.get("question") or "").strip()
        sources = self._sources(args, answer)
        facts = [f for f in answer.get("facts") or [] if isinstance(f, dict) and f.get("label")]

        cleaned = self.clean(
            self.judge(
                self.system_prompt(settings),
                self.user_prompt(question, answer["answer"], sources, settings, facts),
                settings,
                context,
            ),
            settings,
        )
        return {
            "question": question,
            "model": settings["model"],
            "answer_model": answer.get("model"),
            **cleaned,
            "review_flags": self.review(cleaned, settings),
            "citation_coverage": answer.get("citation_coverage"),
            "unknown_citations": answer.get("unknown_citations"),
            "facts": len(facts),
            "provenance": self.provenance(settings, derived_from=[run_id_of(answer)]),
        }

    def _sources(self, args: dict[str, Any], answer: dict[str, Any]) -> list[dict[str, Any]]:
        """The answer's sources in label order, with full text from ``passages`` or the corpus."""
        listed = sorted(
            (s for s in answer.get("sources") or [] if isinstance(s, dict) and s.get("id")),
            key=lambda s: s.get("rank") or 0,
        )
        if not listed:
            raise self.input_error(
                "The answer lists no sources to judge it against.", code="no_candidates"
            )
        given = args.get("passages")
        if given is not None:
            texts = {
                str(p["id"]): p for p in passages_with_text(given, None, tool_name=self.tool_name)
            }
            missing = [s["id"] for s in listed if str(s["id"]) not in texts]
            if missing:
                raise self.input_error(
                    f"passages lack the answer's sources {missing[:5]}.", argument="passages"
                )
            return [{**s, "text": texts[str(s["id"])]["text"]} for s in listed]
        corpus = self.text_arg(args, "corpus") or str(answer.get("corpus") or "").strip()
        if not corpus:
            raise self.input_error(
                "Cannot read the answer's sources: pass corpus='<name>' or passages=[...] with their text.",
                code="missing_required_arguments",
                missing=["corpus"],
            )
        with_text = passages_with_text(
            [{"id": str(s["id"])} for s in listed], corpus, tool_name=self.tool_name
        )
        return [{**s, "text": p["text"]} for s, p in zip(listed, with_text, strict=True)]
