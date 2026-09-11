from anatoolbox.stages import (
    RESEARCH_PIPELINE_STAGES,
    STAGE_ORDER,
    STAGE_PACKAGE_BY_LABEL,
    prefix_to_stage,
    prefixes_for_stage,
    stage_for_tool_name,
)


def test_stage_order_starts_with_prepare_and_ends_with_present():
    assert STAGE_ORDER[0] == "prepare"
    assert STAGE_ORDER[-1] == "present"


def test_pipeline_stages_are_stage_order_without_prepare():
    """The brief gate produces the plan; prepare is not a runner stage."""
    assert RESEARCH_PIPELINE_STAGES == STAGE_ORDER[1:]


def test_stage_labels_map_onto_stage_order():
    assert sorted(STAGE_PACKAGE_BY_LABEL.values()) == sorted(STAGE_ORDER)


def test_stage_for_tool_name_uses_the_prefix():
    assert stage_for_tool_name("retrieve_passages") == "gather"
    assert stage_for_tool_name("aggregate_mentions_by_period") == "analyze"
    assert stage_for_tool_name("tabulate_comparison") == "present"


def test_stage_for_unknown_name_is_none():
    assert stage_for_tool_name("do_stuff") is None
    assert stage_for_tool_name("") is None


def test_longest_prefix_wins():
    """`extract_` and a hypothetical longer sibling must not collide."""
    mapping = prefix_to_stage()
    assert mapping["extract_"] == "extract"
    # 'enrich_' lives in the enrich stage, and so does the stage package name
    assert mapping["enrich_"] == "enrich"


def test_prefixes_for_stage_is_sorted_and_complete():
    gather = prefixes_for_stage("gather")
    assert gather == sorted(gather)
    assert set(gather) == {"retrieve_", "ingest_", "monitor_"}


def test_prefix_to_stage_returns_a_copy():
    """Callers must not be able to corrupt the discovered catalog."""
    prefix_to_stage()["bogus_"] = "nowhere"
    assert "bogus_" not in prefix_to_stage()
