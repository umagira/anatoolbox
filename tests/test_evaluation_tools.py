"""Evaluation tools: draft test questions, retrieval metrics, an LLM judge — and graph facts in answers."""

import json

import pytest

from anatoolbox import ToolContext
from anatoolbox.analyze.calculate.calculate_retrieval_metrics import (
    CalculateRetrievalMetricsTool,
)
from anatoolbox.analyze.score.score_rag_answer import CRITERIA, ScoreRagAnswerTool
from anatoolbox.corpus import LocalCorpus, clear_corpora, register_corpus
from anatoolbox.enrich.synthesize.synthesize_answer import SynthesizeAnswerTool
from anatoolbox.errors import ToolInputError
from anatoolbox.extract.extract.extract_test_questions import ExtractTestQuestionsTool
from anatoolbox.gather.retrieve.retrieve_passages import RetrievePassagesTool

CTX = ToolContext()
FACT = {
    "subject": "Nvidia",
    "relation": "develops",
    "object": "B300",
    "subject_type": "Company",
    "object_type": "Chip",
    "source_ids": ["a1"],
}


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
                    "text": "Nvidia ships the B300 GPU in March.",
                },
                {
                    "id": "a2",
                    "title": "Agents",
                    "date": "2025-06-01",
                    "text": "Google proposes the A2A agent protocol.",
                },
                {"id": "a3", "title": "Tiny", "date": "2025-02-01", "text": "Short."},
            ],
            name="articles",
        )
    )
    yield corpus
    clear_corpora()


# --- extract_test_questions ------------------------------------------------------


def drafted(*questions):
    return json.dumps({"questions": [dict(q) for q in questions]})


class TestExtractTestQuestions:
    def test_questions_keep_their_source_and_skip_unusable_replies(self, fake_llm):
        fake_llm.replies += [
            drafted(
                {"question": "What does this article announce?", "answer": "x", "type": "factual"},
                {
                    "question": "Which GPU does Nvidia ship in March?",
                    "answer": "The B300.",
                    "type": "factual",
                },
            ),
            drafted(),
        ]
        out = ExtractTestQuestionsTool().run(
            {"input": {"corpus": "articles"}, "sample_size": 2, "min_chars": 20}, context=CTX
        )
        assert out["records_sampled"] == 2
        assert len(out["questions"]) == 1 and len(out["skipped_records"]) == 1
        question = out["questions"][0]
        assert question["id"] == "q1" and question["type"] == "factual"
        assert question["reference_answer"] == "The B300."
        assert question["source_ids"][0] in ("a1", "a2") and question["passage_id"] is None
        assert question["title"] in ("Chips", "Agents")
        assert out["types"]["factual"] == 1 and out["types"]["temporal"] == 0
        assert out["model"] == "default-model"  # no evaluation role configured
        assert out["provenance"]["tool"] == "extract_test_questions"

    def test_questions_from_chunks_name_their_article_and_passage(self, fake_llm):
        register_corpus(
            LocalCorpus.from_records(
                [{"id": "a1#2", "source_id": "a1", "text": "Nvidia ships the B300 GPU in March."}],
                name="chunks",
            )
        )
        fake_llm.replies.append(
            drafted(
                {"question": "When does the B300 ship?", "answer": "March.", "type": "temporal"}
            )
        )
        out = ExtractTestQuestionsTool().run(
            {"corpus": "chunks", "sample_size": 1, "min_chars": 0}, context=CTX
        )
        assert out["questions"][0]["source_ids"] == ["a1"]
        assert out["questions"][0]["passage_id"] == "a1#2"

    def test_question_types_shape_the_prompt_and_unknown_types_become_none(self, fake_llm):
        fake_llm.replies.append(
            drafted({"question": "Who ships the B300?", "answer": "Nvidia.", "type": "trivia"})
        )
        out = ExtractTestQuestionsTool().run(
            {
                "corpus": "articles",
                "sample_size": 1,
                "min_chars": 30,
                "question_types": ["temporal", "comparative"],
            },
            context=CTX,
        )
        system = fake_llm.system_prompt()
        assert "- temporal:" in system and "- comparative:" in system
        assert "- factual:" not in system
        assert out["questions"][0]["type"] is None
        assert out["provenance"]["settings"]["question_types"] == ["temporal", "comparative"]

    def test_sampling_is_seeded_and_respects_min_chars(self, fake_llm, articles):
        tool = ExtractTestQuestionsTool()
        settings = tool.settings({"sample_size": 5, "seed": 3, "min_chars": 20})
        first = [r["id"] for r in tool.sample(articles, settings)]
        again = [r["id"] for r in tool.sample(articles, settings)]
        assert first == again and sorted(first) == ["a1", "a2"]  # "Short." is too short

    def test_the_prompt_shows_title_date_and_text(self, fake_llm):
        fake_llm.replies.append(drafted())
        ExtractTestQuestionsTool().run(
            {"corpus": "articles", "sample_size": 1, "min_chars": 0, "seed": 0}, context=CTX
        )
        assert fake_llm.user_prompt().startswith("Title: ")
        assert "Published: 2025-" in fake_llm.user_prompt()

    @pytest.mark.parametrize(
        "args, match",
        [
            ({"questions_per_record": 6}, "at most 5"),
            ({"question_types": ["trivia"]}, "question_types"),
            ({"question_types": []}, "question_types"),
        ],
    )
    def test_bad_arguments(self, fake_llm, args, match):
        with pytest.raises(ToolInputError, match=match):
            ExtractTestQuestionsTool().run({"corpus": "articles", **args}, context=CTX)


# --- calculate_retrieval_metrics ---------------------------------------------------


class TestRetrievalMetrics:
    def test_metrics_for_one_question(self):
        out = CalculateRetrievalMetricsTool().run(
            {"results": [{"relevant": ["b"], "retrieved": ["a", "b", "c"]}], "k": [1, 3]},
            context=CTX,
        )
        row = out["per_question"][0]
        assert row["precision@1"] == 0 and row["hit@1"] == 0
        assert row["precision@3"] == pytest.approx(0.3333)
        assert row["recall@3"] == 1 and row["hit@3"] == 1
        assert row["reciprocal_rank"] == 0.5
        assert out["summary"]["mrr"] == 0.5

    def test_precision_divides_by_k_even_when_fewer_are_returned(self):
        out = CalculateRetrievalMetricsTool().run(
            {"results": [{"relevant": ["a"], "retrieved": ["a"]}], "k": 5}, context=CTX
        )
        assert out["per_question"][0]["precision@5"] == 0.2

    def test_chunks_count_as_their_article_once(self):
        passages = [
            {"id": "a1#0", "source_id": "a1"},
            {"id": "a1#3", "source_id": "a1"},
            {"id": "a2#1", "source_id": "a2"},
        ]
        tool = CalculateRetrievalMetricsTool()
        by_source = tool.run(
            {"results": [{"relevant": ["a2"], "retrieved": passages}], "k": [2]}, context=CTX
        )
        assert by_source["per_question"][0]["hit@2"] == 1
        assert by_source["per_question"][0]["reciprocal_rank"] == 0.5
        by_id = tool.run(
            {"results": [{"relevant": ["a2"], "retrieved": passages}], "k": [2], "match_on": "id"},
            context=CTX,
        )
        assert by_id["per_question"][0]["hit@2"] == 0

    def test_summary_averages_and_provenance_names_the_retrievals(self):
        retrieved = RetrievePassagesTool().run({"query": "GPU", "corpus": "articles"}, context=CTX)
        out = CalculateRetrievalMetricsTool().run(
            {
                "results": [
                    {"question": "GPU?", "relevant": ["a1"], "retrieved": retrieved},
                    {"question": "Agents?", "relevant": ["a2"], "retrieved": ["a3"]},
                ],
                "k": [1],
            },
            context=CTX,
        )
        assert out["summary"]["hit@1"] == 0.5
        assert out["per_question"][0]["question"] == "GPU?"
        assert out["provenance"]["derived_from"] == [retrieved["provenance"]["run_id"]]

    def test_a_question_without_relevant_items_has_no_recall(self):
        out = CalculateRetrievalMetricsTool().run(
            {
                "results": [
                    {"relevant": [], "retrieved": ["a"]},
                    {"relevant": ["a"], "retrieved": ["a"]},
                ],
                "k": [1],
            },
            context=CTX,
        )
        assert out["per_question"][0]["recall@1"] is None
        assert out["summary"]["recall@1"] == 1.0

    def test_subclass_adds_a_metric(self):
        class WithRPrecision(CalculateRetrievalMetricsTool):
            tool_name = "calculate_retrieval_metrics_with_r_precision"

            def metrics(self, retrieved, relevant, settings):
                r = len(relevant)
                found = sum(1 for i in retrieved[:r] if i in relevant)
                return {**super().metrics(retrieved, relevant, settings), "r_precision": found / r}

        out = WithRPrecision().run(
            {"results": [{"relevant": ["a", "b"], "retrieved": ["a", "c", "b"]}], "k": [1]},
            context=CTX,
        )
        assert out["summary"]["r_precision"] == 0.5

    @pytest.mark.parametrize(
        "args, match",
        [
            ({"results": []}, "non-empty"),
            ({"results": [{"relevant": ["a"], "retrieved": "a"}]}, "retrieved"),
            ({"results": [{"retrieved": ["a"]}]}, "relevant"),
            ({"results": [{"relevant": ["a"], "retrieved": ["a"]}], "k": [0]}, "k must"),
        ],
    )
    def test_bad_input(self, args, match):
        with pytest.raises(ToolInputError, match=match):
            CalculateRetrievalMetricsTool().run(args, context=CTX)


# --- graph facts in answers --------------------------------------------------------


@pytest.fixture
def retrieved():
    return RetrievePassagesTool().run({"query": "Nvidia GPU", "corpus": "articles"}, context=CTX)


class TestAnswersWithGraphFacts:
    def test_facts_are_labelled_shown_and_their_citations_checked(self, fake_llm, retrieved):
        fake_llm.replies.append("Nvidia develops the B300 [S1][G1]. It also makes cars [G3].")
        graph_result = {"facts": [FACT, dict(FACT)], "provenance": {"run_id": "retrieve_facts-1"}}
        out = SynthesizeAnswerTool().run(
            {"question": "What does Nvidia ship?", "input": retrieved, "facts": graph_result},
            context=CTX,
        )
        user = fake_llm.user_prompt()
        assert (
            "Graph facts from the knowledge graph:\n\n"
            "[G1] Nvidia [Company] --develops--> B300 [Chip]" in user
        )
        assert user.endswith("citing sources and graph facts by label.")
        assert [f["label"] for f in out["facts"]] == ["G1"]  # the repeated fact was dropped
        assert out["facts"][0]["source_ids"] == ["a1"]
        assert out["cited"] == ["S1", "G1"] and out["unknown_citations"] == ["G3"]
        assert out["uncited_facts"] == []
        assert out["provenance"]["settings"]["facts"] == 1
        assert out["provenance"]["derived_from"] == [
            retrieved["provenance"]["run_id"],
            "retrieve_facts-1",
        ]

    def test_without_facts_the_prompt_is_text_only(self, fake_llm, retrieved):
        fake_llm.replies.append("It ships [S1].")
        out = SynthesizeAnswerTool().run(
            {"question": "What ships?", "input": retrieved}, context=CTX
        )
        assert "Graph facts" not in fake_llm.user_prompt()
        assert fake_llm.user_prompt().endswith("citing sources by label.")
        assert out["facts"] == [] and out["uncited_facts"] == []

    def test_max_facts_and_bad_facts(self, fake_llm, retrieved):
        facts = [{**FACT, "object": f"B{n}"} for n in range(5)]
        fake_llm.replies.append("Yes [S1].")
        out = SynthesizeAnswerTool().run(
            {"question": "q", "input": retrieved, "facts": facts, "max_facts": 2}, context=CTX
        )
        assert [f["object"] for f in out["facts"]] == ["B0", "B1"]
        assert out["uncited_facts"] == ["G1", "G2"]
        with pytest.raises(ToolInputError, match="subject"):
            SynthesizeAnswerTool().run(
                {"question": "q", "input": retrieved, "facts": [{"subject": "Nvidia"}]},
                context=CTX,
            )


# --- score_rag_answer ------------------------------------------------------------


def judged(scores, claims=()):
    return json.dumps(
        {
            "scores": [
                {"criterion": c, "score": s, "rationale": f"{c} ok"} for c, s in scores.items()
            ],
            "unsupported_claims": list(claims),
        }
    )


@pytest.fixture
def answer(fake_llm, retrieved):
    fake_llm.replies.append("Nvidia ships the B300 [S1].")
    return SynthesizeAnswerTool().run(
        {"question": "What does Nvidia ship?", "input": retrieved}, context=CTX
    )


class TestScoreRagAnswer:
    def test_scores_rationales_and_claims(self, fake_llm, answer):
        assert answer["corpus"] == "articles"
        fake_llm.replies.append(
            judged(
                {
                    "faithfulness": 5,
                    "relevance": 4,
                    "context_grounding": 4,
                    "temporal_grounding": None,
                },
                claims=["in March"],
            )
        )
        out = ScoreRagAnswerTool().run({"answer": answer}, context=CTX)
        assert out["scores"] == {
            "faithfulness": 5,
            "relevance": 4,
            "context_grounding": 4,
            "temporal_grounding": None,  # does not apply: the question is not about time
        }
        assert out["rationales"]["temporal_grounding"] == "temporal_grounding ok"
        assert out["unsupported_claims"] == ["in March"]
        assert out["citation_coverage"] == answer["citation_coverage"]
        assert out["provenance"]["derived_from"] == [answer["provenance"]["run_id"]]

    def test_the_judge_sees_full_source_texts_and_the_criteria(self, fake_llm, answer):
        fake_llm.replies.append(judged({"faithfulness": 5}))
        ScoreRagAnswerTool().run(
            {"answer": answer, "reference_answer": "The B300 GPU."}, context=CTX
        )
        user, system = fake_llm.user_prompt(1), fake_llm.system_prompt(1)
        assert "[S1] Chips · 2025-01-10\nNvidia ships the B300 GPU in March." in user
        assert "Reference answer: The B300 GPU." in user
        assert user.rstrip().endswith("Nvidia ships the B300 [S1].")
        for criterion in ("faithfulness", "relevance", "context_grounding", "temporal_grounding"):
            assert f"- {criterion}:" in system
        assert "- correctness:" in system  # added because a reference answer was given
        assert "Graph facts" not in user

    def test_the_judge_sees_the_graph_facts_the_answer_used(self, fake_llm, retrieved):
        fake_llm.replies.append("Nvidia develops the B300 [G1].")
        with_facts = SynthesizeAnswerTool().run(
            {"question": "What does Nvidia develop?", "input": retrieved, "facts": [FACT]},
            context=CTX,
        )
        fake_llm.replies.append(judged({"faithfulness": 5}))
        out = ScoreRagAnswerTool().run(
            {"answer": with_facts, "criteria": ["faithfulness"]}, context=CTX
        )
        assert "Graph facts:\n\n[G1] Nvidia [Company] --develops--> B300 [Chip]" in (
            fake_llm.user_prompt(1)
        )
        assert out["facts"] == 1

    def test_scores_are_clamped_and_missing_ones_are_none(self, fake_llm, answer):
        fake_llm.replies.append(
            json.dumps(
                {"scores": {"faithfulness": 9, "relevance": "n/a"}, "unsupported_claims": []}
            )
        )
        out = ScoreRagAnswerTool().run(
            {"answer": answer, "criteria": ["faithfulness", "relevance", "context_grounding"]},
            context=CTX,
        )
        assert out["scores"] == {"faithfulness": 5, "relevance": None, "context_grounding": None}

    def test_explicit_passages_when_the_answer_names_no_corpus(self, fake_llm):
        fake_llm.replies.append("It ships [S1].")
        passages = [{"id": "p1", "text": "Nvidia ships the B300."}]
        result = SynthesizeAnswerTool().run(
            {"question": "What ships?", "passages": passages}, context=CTX
        )
        with pytest.raises(ToolInputError, match="corpus"):
            ScoreRagAnswerTool().run({"answer": result}, context=CTX)
        fake_llm.replies.append(judged({"faithfulness": 4}))
        out = ScoreRagAnswerTool().run({"answer": result, "passages": passages}, context=CTX)
        assert out["scores"]["faithfulness"] == 4

    @pytest.mark.parametrize(
        "args, match",
        [
            ({"criteria": ["correctness"]}, "reference_answer"),
            ({"criteria": ["groundedness"]}, "Unknown criteria"),
            ({"criteria": []}, "non-empty"),
        ],
    )
    def test_bad_criteria(self, fake_llm, answer, args, match):
        with pytest.raises(ToolInputError, match=match):
            ScoreRagAnswerTool().run({"answer": answer, **args}, context=CTX)

    def test_not_an_answer(self, fake_llm):
        with pytest.raises(ToolInputError, match="synthesize_answer"):
            ScoreRagAnswerTool().run({"answer": "just text"}, context=CTX)

    def test_subclass_adds_a_criterion(self, fake_llm, answer):
        class ScoreWithConciseness(ScoreRagAnswerTool):
            tool_name = "score_rag_answer_with_conciseness"
            criteria = {
                **CRITERIA,
                "conciseness": "The answer has no filler. 5: none. 1: mostly filler.",
            }

        fake_llm.replies.append(judged({"conciseness": 2}))
        out = ScoreWithConciseness().run(
            {"answer": answer, "criteria": ["conciseness"]}, context=CTX
        )
        assert out["scores"] == {"conciseness": 2}
        assert "- conciseness: The answer has no filler." in fake_llm.system_prompt(1)

    def test_review_flags_catch_a_judge_that_contradicts_itself(self, fake_llm, answer):
        copied = CRITERIA["faithfulness"].split(" 5:")[0]  # the description, not a reason
        fake_llm.replies.append(
            json.dumps(
                {
                    "scores": [
                        {"criterion": "faithfulness", "score": 5, "rationale": copied},
                        {"criterion": "relevance", "score": 4, "rationale": "It names the chip."},
                    ],
                    "unsupported_claims": ["in March"],
                }
            )
        )
        out = ScoreRagAnswerTool().run(
            {"answer": answer, "criteria": ["faithfulness", "relevance", "context_grounding"]},
            context=CTX,
        )
        assert out["review_flags"] == [
            "faithfulness: the rationale repeats the criterion instead of explaining the score",
            "context_grounding: no score and no rationale",
            "faithfulness: top score, yet unsupported claims are listed",
        ]

    def test_a_consistent_judgment_has_no_flags(self, fake_llm, answer):
        fake_llm.replies.append(judged({"faithfulness": 4, "relevance": 5}))
        out = ScoreRagAnswerTool().run(
            {"answer": answer, "criteria": ["faithfulness", "relevance"]}, context=CTX
        )
        assert out["review_flags"] == []
