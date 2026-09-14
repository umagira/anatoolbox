"""synthesize_answer: curation, prompting, citation checks, provenance."""

import json
import re

import pytest

from anatoolbox import ToolContext
from anatoolbox.corpus import LocalCorpus, clear_corpora, local_ref, register_corpus
from anatoolbox.enrich.synthesize.synthesize_answer import (
    SynthesizeAnswerTool,
    citation_coverage,
    curate,
    extract_citations,
    link_citations,
)
from anatoolbox.errors import ToolInputError
from anatoolbox.memory.facade import Memory
from anatoolbox.memory.policy import DefaultMemoryPolicy
from anatoolbox.memory.store import InMemoryRecordsetStore

DOCS = [
    {
        "id": "d1",
        "title": "MCP adoption grows",
        "date": "2025-06-20",
        "url": "https://example.org/mcp",
        "text": "Several model providers announced support for the Model Context Protocol. "
        + "Detail. " * 300
        + "Late fact: 40 servers.",
    },
    {
        "id": "d2",
        "title": "Chip rules tighten",
        "date": "2025-04-03",
        "url": "https://example.org/chips",
        "text": "Export controls restrict advanced AI chips.",
    },
    {
        "id": "d3",
        "title": "Agents in browsers",
        "date": "2025-02-11",
        "text": "Browser agents complete multi-step tasks.",
    },
]


@pytest.fixture(autouse=True)
def clean():
    clear_corpora()
    yield
    clear_corpora()


@pytest.fixture
def ctx():
    memory = Memory(InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="p", session_id="s")
    register_corpus(LocalCorpus.from_records(DOCS, name="news"))
    memory.remember(
        object_type="passages",
        stage="gather",
        produced_by="retrieve_passages",
        args={"query": "what is new"},
        ref=local_ref(corpus="news", ids=["d1", "d2", "d3"]),
        count=3,
        summary="3 passages",
    )
    return ToolContext(project="p", session_id="s", recordsets=memory)


def bare():
    return ToolContext(project="p", session_id="s")


def explicit(n):
    return [{"id": f"p{i}", "text": f"passage number {i}"} for i in range(1, n + 1)]


def prompt_label_order(prompt):
    return re.findall(r"^\[(S\d+)\]", prompt, flags=re.MULTILINE)


class TestAnswering:
    def test_citations_are_resolved_and_checked(self, fake_llm, ctx):
        fake_llm.replies = [
            ["MCP support is spreading [S1]. ", "Chip exports are restricted [S2]."]
        ]
        out = SynthesizeAnswerTool().run({"question": "What is new?"}, context=ctx)
        assert out["cited"] == ["S1", "S2"]
        assert out["unknown_citations"] == [] and out["uncited_sources"] == ["S3"]
        assert "[S1](https://example.org/mcp)" in out["answer_markdown"]

    def test_the_strong_role_model_is_used_and_recorded(self, fake_llm, ctx):
        fake_llm.replies = [["ok [S1]"]]
        out = SynthesizeAnswerTool().run({"question": "q"}, context=ctx)
        assert fake_llm.calls[0]["model"] == "strong-model" == out["model"]

    def test_an_explicit_model_wins(self, fake_llm, ctx):
        fake_llm.replies = [["ok"]]
        out = SynthesizeAnswerTool().run({"question": "q", "model": "other"}, context=ctx)
        assert fake_llm.calls[0]["model"] == "other" == out["model"]

    def test_invented_labels_are_reported_not_linked(self, fake_llm, ctx):
        fake_llm.replies = [["Both matter [S1, S3]. Also [S7]."]]
        out = SynthesizeAnswerTool().run({"question": "q"}, context=ctx)
        assert out["cited"] == ["S1", "S3"] and out["unknown_citations"] == ["S7"]
        assert "[S1](https://example.org/mcp)[S3]" in out["answer_markdown"], "S3 has no url"

    def test_a_reasoning_block_is_not_part_of_the_answer(self, fake_llm, ctx):
        fake_llm.replies = [["<think>maybe [S9]</think>", "Answer [S1]."]]
        out = SynthesizeAnswerTool().run({"question": "q"}, context=ctx)
        assert out["answer"] == "Answer [S1]." and out["unknown_citations"] == []

    def test_the_prompt_shows_labels_dates_and_full_text(self, fake_llm, ctx):
        fake_llm.replies = [["ok"]]
        SynthesizeAnswerTool().run(
            {"question": "What is new?", "max_chars_per_passage": 5000}, context=ctx
        )
        prompt = fake_llm.user_prompt()
        assert "[S1] MCP adoption grows · 2025-06-20 · https://example.org/mcp" in prompt
        assert "Late fact: 40 servers." in prompt, "full text from the corpus, not a snippet"
        assert prompt.startswith("Question: What is new?")

    def test_long_passages_are_truncated(self, fake_llm, ctx):
        fake_llm.replies = [["ok"]]
        SynthesizeAnswerTool().run({"question": "q", "max_chars_per_passage": 100}, context=ctx)
        assert "Late fact" not in fake_llm.user_prompt() and " …" in fake_llm.user_prompt()

    def test_modes_change_the_instructions(self, fake_llm, ctx):
        fake_llm.replies = [["ok"], ["ok"]]
        SynthesizeAnswerTool().run({"question": "q"}, context=ctx)
        SynthesizeAnswerTool().run({"question": "q", "mode": "blended"}, context=ctx)
        assert "say so plainly" in fake_llm.system_prompt(
            0
        ) and "background" not in fake_llm.system_prompt(0)
        assert "background" in fake_llm.system_prompt(1)

    def test_instructions_reach_the_prompt(self, fake_llm, ctx):
        fake_llm.replies = [["ok"]]
        SynthesizeAnswerTool().run(
            {"question": "q", "instructions": "Answer in two sentences."}, context=ctx
        )
        assert "Additional instructions: Answer in two sentences." in fake_llm.user_prompt()


class TestCuration:
    def test_edges_order_puts_the_best_passages_first_and_last(self, fake_llm):
        fake_llm.replies = [["ok"]]
        SynthesizeAnswerTool().run({"question": "q", "passages": explicit(5)}, context=bare())
        assert prompt_label_order(fake_llm.user_prompt()) == ["S1", "S3", "S5", "S4", "S2"]

    def test_relevance_order_keeps_rank_order(self, fake_llm):
        fake_llm.replies = [["ok"]]
        SynthesizeAnswerTool().run(
            {"question": "q", "passages": explicit(5), "order": "relevance"}, context=bare()
        )
        assert prompt_label_order(fake_llm.user_prompt()) == ["S1", "S2", "S3", "S4", "S5"]

    def test_duplicates_are_dropped_before_prompting(self, fake_llm):
        fake_llm.replies = [["ok"]]
        passages = [
            {"id": "a", "text": "Same  Text"},
            {"id": "b", "text": "same text"},
            {"id": "c", "text": "other"},
        ]
        out = SynthesizeAnswerTool().run({"question": "q", "passages": passages}, context=bare())
        assert out["dropped_duplicates"] == 1
        assert [s["id"] for s in out["sources"]] == ["a", "c"]

    def test_max_passages_caps_the_sources(self, fake_llm):
        fake_llm.replies = [["ok"]]
        out = SynthesizeAnswerTool().run(
            {"question": "q", "passages": explicit(5), "max_passages": 2}, context=bare()
        )
        assert [s["label"] for s in out["sources"]] == ["S1", "S2"]

    def test_curate_reports_rank_and_position(self):
        sources, _ = curate(explicit(3), order="edges")
        assert [(s["label"], s["rank"], s["position"]) for s in sources] == [
            ("S1", 1, 1),
            ("S3", 3, 2),
            ("S2", 2, 3),
        ]


class TestCitationHelpers:
    def test_extract_citations_accepts_common_forms(self):
        assert extract_citations("a [S2] b [S1, S3] c [S2][S4] d [ S5 ; S6 ]") == [
            "S2",
            "S1",
            "S3",
            "S4",
            "S5",
            "S6",
        ]

    def test_link_citations_only_links_known_urls(self):
        linked = link_citations("x [S1, S2]", {"S1": {"url": "https://a"}, "S2": {}})
        assert linked == "x [S1](https://a)[S2]"


class TestProvenanceAndErrors:
    def test_lineage_and_provenance(self, fake_llm, ctx):
        fake_llm.replies = [["ok [S2]"]]
        out = SynthesizeAnswerTool().run({"question": "q", "mode": "blended"}, context=ctx)
        record = ctx.recordsets.get(out["handle"])
        assert record.object_type == "answer" and record.derived_from == ["passages_1"]
        assert record.ref["mode"] == "value"
        assert (record.args["model"], record.args["mode"]) == ("strong-model", "blended")

    def test_execute_renders_the_answer_with_its_sources(self, fake_llm, ctx):
        fake_llm.replies = [["ok [S1]"]]
        payload = json.loads(SynthesizeAnswerTool().execute({"question": "q"}, context=ctx))
        assert payload["render"]["render_type"] == "markdown"
        assert "**Sources**" in payload["render"]["content"]
        assert (
            "MCP adoption grows · 2025-06-20 — https://example.org/mcp"
            in payload["render"]["content"]
        )

    @pytest.mark.parametrize(
        "bad, match",
        [
            ({}, "question"),
            ({"question": "q", "mode": "creative"}, "mode"),
            ({"question": "q", "order": "random"}, "order"),
            ({"question": "q", "max_passages": 0}, "max_passages"),
        ],
    )
    def test_invalid_arguments(self, fake_llm, ctx, bad, match):
        with pytest.raises(ToolInputError, match=match):
            SynthesizeAnswerTool().run(bad, context=ctx)

    def test_no_passages_and_no_memory_is_explained(self, fake_llm):
        with pytest.raises(ToolInputError, match="retrieve_passages"):
            SynthesizeAnswerTool().run({"question": "q"}, context=bare())

    def test_passages_without_text_are_explained(self, fake_llm):
        with pytest.raises(ToolInputError, match="no passages with text"):
            SynthesizeAnswerTool().run(
                {"question": "q", "passages": [{"id": "a", "text": "  "}]}, context=bare()
            )

    def test_malformed_passages_are_rejected(self, fake_llm):
        with pytest.raises(ToolInputError, match="'id' and 'text'"):
            SynthesizeAnswerTool().run({"question": "q", "passages": [{"id": "a"}]}, context=bare())


class TestCitationCoverage:
    def test_counts_sentences_with_inline_citations(self):
        assert citation_coverage("Agents browse [S1]. Nothing here. Chips are scarce [S2].") == {
            "statements": 3,
            "statements_with_citations": 2,
            "citation_coverage": 0.667,
        }

    def test_a_citation_after_the_period_counts_for_that_sentence(self):
        assert citation_coverage("Chips are restricted. [S2]")["statements_with_citations"] == 1

    def test_sentences_ending_in_a_closing_quote_are_split(self):
        result = citation_coverage('She said "agents are risky [S1]." Then nothing more.')
        assert (result["statements"], result["statements_with_citations"]) == (2, 1)

    def test_citations_dumped_on_their_own_line_count_for_nothing(self):
        """Seen with Qwen3-0.6B: every label on a final line, no sentence attributed."""
        result = citation_coverage("First claim. Second claim.\n\n[S2][S4][S5]")
        assert (result["statements"], result["citation_coverage"]) == (2, 0.0)

    def test_headings_are_not_statements(self):
        assert citation_coverage("## Findings\nAgents browse [S1].")["statements"] == 1

    def test_an_empty_answer_has_no_coverage(self):
        assert citation_coverage("")["citation_coverage"] is None

    def test_the_tool_reports_coverage_and_asks_for_inline_citations(self, fake_llm, ctx):
        fake_llm.replies = [["One [S1]. Two."]]
        out = SynthesizeAnswerTool().run({"question": "q"}, context=ctx)
        assert out["citation_coverage"] == 0.5
        assert "Do not collect citations at the end" in fake_llm.system_prompt()
        assert (
            ctx.recordsets.records(ctx.recordsets.get(out["handle"]))[0]["citation_coverage"] == 0.5
        )
