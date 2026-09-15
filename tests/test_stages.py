"""The stage taxonomy is a product decision. Pin all of it."""

from anatoolbox.stages import (
    RESEARCH_PIPELINE_STAGES,
    STAGE_DESCRIPTIONS,
    STAGE_ORDER,
    STAGE_PACKAGE_BY_LABEL,
    prefix_to_stage,
    prefixes_for_stage,
    stage_for_tool_name,
)

EXPECTED_PREFIXES = {
    "prepare": {"plan_"},
    "gather": {"ingest_", "retrieve_", "rerank_", "rewrite_"},
    "preprocess": {"chunk_", "clean_", "deduplicate_", "normalize_", "parse_"},
    "extract": {"detect_", "extract_", "resolve_"},
    "analyze": {"aggregate_", "calculate_", "classify_", "compare_", "score_"},
    "enrich": {
        "assess_",
        "benchmark_",
        "enrich_",
        "explain_",
        "interpret_",
        "recommend_",
        "synthesize_",
        "validate_",
    },
}

REMOVED_PREFIXES = [
    "monitor_",
    "identify_",
    "model_",
    "trend_",
    "calibrate_",
    "weight_",
    "chart_",
    "export_",
    "report_",
    "summarize_",
    "tabulate_",
    "visualize_",
]


def test_stage_order():
    assert STAGE_ORDER == ["prepare", "gather", "preprocess", "extract", "analyze", "enrich"]


def test_presentation_is_not_a_stage():
    """Presentation belongs in the frontend; tools only declare a render_type."""
    assert "present" not in STAGE_ORDER
    assert "present" not in STAGE_PACKAGE_BY_LABEL.values()


def test_pipeline_stages_are_stage_order_without_prepare():
    """Prepare produces the plan; it is not a step of the evidence pipeline."""
    assert RESEARCH_PIPELINE_STAGES == STAGE_ORDER[1:]


def test_stage_labels_map_onto_stage_order():
    assert sorted(STAGE_PACKAGE_BY_LABEL.values()) == sorted(STAGE_ORDER)


def test_every_stage_is_described():
    assert set(STAGE_DESCRIPTIONS) == set(STAGE_ORDER)
    assert all(text.strip() for text in STAGE_DESCRIPTIONS.values())


def test_prefixes_per_stage_are_exactly_the_taxonomy():
    for stage, expected in EXPECTED_PREFIXES.items():
        assert set(prefixes_for_stage(stage)) == expected, stage
    assert len(prefix_to_stage()) == sum(len(p) for p in EXPECTED_PREFIXES.values()) == 26


def test_prefixes_for_stage_is_sorted():
    for stage in STAGE_ORDER:
        prefixes = prefixes_for_stage(stage)
        assert prefixes == sorted(prefixes)


def test_stage_for_tool_name_uses_the_prefix():
    assert stage_for_tool_name("retrieve_passages") == "gather"
    assert stage_for_tool_name("rerank_passages") == "gather"
    assert stage_for_tool_name("rewrite_query_for_retrieval") == "gather"
    assert stage_for_tool_name("clean_boilerplate") == "preprocess"
    assert stage_for_tool_name("parse_pdf_report") == "preprocess"
    assert stage_for_tool_name("deduplicate_articles") == "preprocess"
    assert stage_for_tool_name("chunk_by_size") == "preprocess"
    assert stage_for_tool_name("extract_entities") == "extract"
    assert stage_for_tool_name("aggregate_mentions_by_period") == "analyze"
    assert stage_for_tool_name("synthesize_answer") == "enrich"


def test_removed_prefixes_no_longer_name_a_stage():
    for prefix in REMOVED_PREFIXES:
        assert stage_for_tool_name(f"{prefix}something") is None, prefix


def test_stage_for_unknown_name_is_none():
    assert stage_for_tool_name("do_stuff") is None
    assert stage_for_tool_name("") is None


def test_longest_prefix_wins():
    """`extract_` and a hypothetical longer sibling must not collide."""
    mapping = prefix_to_stage()
    assert mapping["extract_"] == "extract"
    # 'enrich_' lives in the enrich stage, and so does the stage package name
    assert mapping["enrich_"] == "enrich"


def test_prefix_to_stage_returns_a_copy():
    """Callers must not be able to corrupt the discovered catalog."""
    prefix_to_stage()["bogus_"] = "nowhere"
    assert "bogus_" not in prefix_to_stage()
