"""rewrite_query_for_retrieval: prompting, reply cleaning, provenance."""

import json

import pytest

from anatoolbox import ToolContext
from anatoolbox.errors import ToolInputError
from anatoolbox.gather.rewrite.rewrite_query_for_retrieval import (
    RewriteQueryForRetrievalTool,
    clean_reply,
    question_names_a_period,
)
from anatoolbox.memory.facade import Memory
from anatoolbox.memory.policy import DefaultMemoryPolicy
from anatoolbox.memory.store import InMemoryRecordsetStore

QUESTION = "What changed for AI agents in 2025?"


def reply(queries, terms=(), start=None, end=None):
    return json.dumps(
        {
            "queries": list(queries),
            "exact_terms": list(terms),
            "time_range": {"from": start, "to": end},
        }
    )


def bare():
    return ToolContext(project="p", session_id="s")


def test_queries_are_cleaned_and_deduplicated(fake_llm):
    fake_llm.replies = [
        reply(
            [
                {"query": "AI agent launches 2025", "purpose": "p"},
                {"query": "ai AGENT launches 2025", "purpose": "dup"},
                {"query": "agent standards MCP", "purpose": "p2"},
            ],
            terms=["MCP", "MCP", " "],
            start="2025-01-01",
        )
    ]
    out = RewriteQueryForRetrievalTool().run({"question": QUESTION}, context=bare())
    assert out["query_texts"] == ["AI agent launches 2025", "agent standards MCP"]
    assert out["exact_terms"] == ["MCP"]
    assert out["time_range"] == {"from": "2025-01-01", "to": None}
    assert out["fallback"] is False


def test_the_fast_role_model_is_used(fake_llm):
    fake_llm.replies = [reply([{"query": "a", "purpose": ""}])]
    out = RewriteQueryForRetrievalTool().run({"question": QUESTION}, context=bare())
    assert fake_llm.calls[0]["model"] == "fast-model" == out["model"]


def test_a_strict_schema_is_requested(fake_llm):
    fake_llm.replies = [reply([{"query": "a", "purpose": ""}])]
    RewriteQueryForRetrievalTool().run({"question": QUESTION}, context=bare())
    assert fake_llm.calls[0]["response_format"]["type"] == "json_schema"


def test_max_queries_caps_the_output(fake_llm):
    fake_llm.replies = [reply([{"query": f"q{i}", "purpose": ""} for i in range(5)])]
    out = RewriteQueryForRetrievalTool().run(
        {"question": QUESTION, "max_queries": 2}, context=bare()
    )
    assert out["query_texts"] == ["q0", "q1"]


@pytest.mark.parametrize(
    "strategy, phrase",
    [("expand", "different phrasings"), ("decompose", "several parts"), ("clarify", "one precise")],
)
def test_strategies_change_the_instructions(fake_llm, strategy, phrase):
    fake_llm.replies = [reply([{"query": "a", "purpose": ""}])]
    RewriteQueryForRetrievalTool().run({"question": QUESTION, "strategy": strategy}, context=bare())
    assert phrase in fake_llm.system_prompt()


def test_clarify_keeps_a_single_query(fake_llm):
    fake_llm.replies = [reply([{"query": "a", "purpose": ""}, {"query": "b", "purpose": ""}])]
    out = RewriteQueryForRetrievalTool().run(
        {"question": QUESTION, "strategy": "clarify"}, context=bare()
    )
    assert out["query_texts"] == ["a"]


def test_the_collection_description_reaches_the_prompt(fake_llm):
    fake_llm.replies = [reply([{"query": "a", "purpose": ""}])]
    RewriteQueryForRetrievalTool().run(
        {"question": QUESTION, "collection": "AI news, 2024-2025"}, context=bare()
    )
    assert "The collection covers: AI news, 2024-2025" in fake_llm.system_prompt()


def test_an_empty_reply_falls_back_to_the_question(fake_llm):
    fake_llm.replies = [reply([])]
    out = RewriteQueryForRetrievalTool().run({"question": QUESTION}, context=bare())
    assert out["query_texts"] == [QUESTION] and out["fallback"] is True


def test_clean_reply_tolerates_plain_strings_bad_dates_and_junk():
    cleaned = clean_reply(
        {
            "queries": ["a", "", 3, "b"],
            "exact_terms": None,
            "time_range": {"from": "last spring", "to": "2025-06-30T00:00"},
        },
        "What happened by June 2025?",
        5,
    )
    assert [q["query"] for q in cleaned["queries"]] == ["a", "b"]
    assert cleaned["exact_terms"] == []
    assert cleaned["time_range"] == {"from": None, "to": "2025-06-30"}
    assert clean_reply("not a dict", "question", 3)["fallback"] is True


def test_provenance(fake_llm):
    memory = Memory(InMemoryRecordsetStore(), DefaultMemoryPolicy(), project="p", session_id="s")
    fake_llm.replies = [reply([{"query": "a", "purpose": "x"}])]
    out = RewriteQueryForRetrievalTool().run(
        {"question": QUESTION, "strategy": "decompose"},
        context=ToolContext(project="p", session_id="s", recordsets=memory),
    )
    record = memory.get(out["handle"])
    assert record.object_type == "queries" and record.ref["mode"] == "value"
    assert (record.args["strategy"], record.args["model"]) == ("decompose", "fast-model")


@pytest.mark.parametrize(
    "bad, match",
    [
        ({}, "question"),
        ({"question": "q", "strategy": "shuffle"}, "strategy"),
        ({"question": "q", "max_queries": 0}, "max_queries"),
        ({"question": "q", "max_queries": 9}, "max_queries"),
    ],
)
def test_invalid_arguments(fake_llm, bad, match):
    with pytest.raises(ToolInputError, match=match):
        RewriteQueryForRetrievalTool().run(bad, context=bare())


def test_an_inverted_time_range_is_discarded():
    """Seen with Qwen3-0.6B: it copied the collection's period, backwards."""
    cleaned = clean_reply(
        {
            "queries": ["a"],
            "exact_terms": [],
            "time_range": {"from": "2024-09-01", "to": "2024-08-31"},
        },
        "question",
        3,
    )
    assert cleaned["time_range"] == {"from": None, "to": None}


def test_the_prompt_rules_out_the_collection_period(fake_llm):
    fake_llm.replies = [reply([{"query": "a", "purpose": ""}])]
    RewriteQueryForRetrievalTool().run(
        {"question": QUESTION, "collection": "AI news, 2024-2025"}, context=bare()
    )
    assert "never the period the collection covers" in fake_llm.system_prompt()


def test_a_time_range_the_question_does_not_name_is_dropped():
    """Seen with Qwen3-0.6B (expand): it filled in the collection's span, 2024-09-01 to 2025-08-31."""
    cleaned = clean_reply(
        {
            "queries": ["a"],
            "exact_terms": [],
            "time_range": {"from": "2024-09-01", "to": "2025-08-31"},
        },
        "What security risks do autonomous browser agents create?",
        3,
    )
    assert cleaned["time_range"] == {"from": None, "to": None}


@pytest.mark.parametrize(
    "question, names_a_period",
    [
        ("What changed for AI agents in 2025?", True),
        ("How did chip sales do in Q3?", True),
        ("What happened since March?", True),
        ("Which models appeared in the last 6 months?", True),
        ("What risks may agents create?", False),
        ("How are companies responding?", False),
    ],
)
def test_question_names_a_period(question, names_a_period):
    assert question_names_a_period(question) is names_a_period
