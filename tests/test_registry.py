"""The registry is the package's extension point — it had no coverage before."""

import pytest
from conftest import make_tool

from anatoolbox import ToolSchema
from anatoolbox.registry import (
    ToolRegistrationError,
    clear_registry,
    register_tool,
    registered_names,
    registry_names,
    resolve_tools,
    tools_by_stage,
    unregister_tool,
)


def test_registry_starts_empty():
    """This package ships contracts, not a tool catalog."""
    clear_registry()
    assert registered_names() == []


def test_register_and_resolve_round_trip():
    register_tool(make_tool("retrieve_passages"))
    assert registered_names() == ["retrieve_passages"]
    assert resolve_tools(["retrieve_passages"])[0].schema.name == "retrieve_passages"


def test_resolve_preserves_order():
    for n in ("retrieve_passages", "score_rag_answers", "clean_boilerplate"):
        register_tool(make_tool(n))
    got = [t.schema.name for t in resolve_tools(["clean_boilerplate", "retrieve_passages"])]
    assert got == ["clean_boilerplate", "retrieve_passages"]


def test_unknown_tool_name_lists_what_is_registered():
    register_tool(make_tool("retrieve_passages"))
    with pytest.raises(KeyError, match="retrieve_passages"):
        resolve_tools(["retrieve_nope"])


def test_unknown_registry_is_reported():
    with pytest.raises(KeyError, match="Unknown tool registry"):
        resolve_tools(["anything"], registry="does_not_exist")


def test_duplicate_registration_rejected_unless_replace():
    register_tool(make_tool("retrieve_passages"))
    with pytest.raises(ToolRegistrationError, match="already registered"):
        register_tool(make_tool("retrieve_passages"))
    replacement = make_tool("retrieve_passages", result={"v": 2})
    register_tool(replacement, replace=True)
    assert resolve_tools(["retrieve_passages"])[0] is replacement


def test_unknown_prefix_rejected():
    """The grammar is the product: a tool must name a task type."""
    with pytest.raises(ToolRegistrationError, match="task-type prefix"):
        register_tool(make_tool("do_stuff"))


def test_unknown_prefix_can_be_bypassed_explicitly():
    register_tool(make_tool("do_stuff"), allow_unknown_prefix=True)
    assert "do_stuff" in registered_names()


@pytest.mark.parametrize(
    "name, missing",
    [("retrieve_a", "run"), ("retrieve_b", "execute")],
)
def test_dual_use_contract_enforced(name, missing):
    tool = make_tool(name)
    delattr(type(tool), missing)
    with pytest.raises(ToolRegistrationError, match=missing):
        register_tool(tool)


def test_schema_must_be_present_and_well_formed():
    class NoSchema:
        def run(self, args, *, context):
            return {}

        def execute(self, args, *, context):
            return ""

    with pytest.raises(ToolRegistrationError, match="no `schema`"):
        register_tool(NoSchema())

    class BadInputSchema:
        schema = ToolSchema(name="retrieve_x", description="d", input_schema=None)

        def run(self, args, *, context):
            return {}

        def execute(self, args, *, context):
            return ""

    with pytest.raises(ToolRegistrationError, match="input_schema"):
        register_tool(BadInputSchema())


def test_named_registries_are_isolated():
    register_tool(make_tool("retrieve_passages"))
    register_tool(make_tool("retrieve_reports"), registry="esg")
    assert registered_names() == ["retrieve_passages"]
    assert registered_names("esg") == ["retrieve_reports"]
    assert "esg" in registry_names()


def test_unregister_is_idempotent():
    register_tool(make_tool("retrieve_passages"))
    unregister_tool("retrieve_passages")
    unregister_tool("retrieve_passages")
    assert registered_names() == []


def test_tools_grouped_by_stage():
    register_tool(make_tool("retrieve_passages"))
    register_tool(make_tool("score_rag_answers"))
    register_tool(make_tool("clean_boilerplate"))
    assert tools_by_stage() == {
        "gather": ["retrieve_passages"],
        "analyze": ["score_rag_answers"],
        "preprocess": ["clean_boilerplate"],
    }


def test_tool_registry_alias_stays_live():
    """`from anatoolbox import TOOL_REGISTRY` must not go stale on registration."""
    from anatoolbox import TOOL_REGISTRY

    register_tool(make_tool("retrieve_passages"))
    assert "retrieve_passages" in TOOL_REGISTRY
