import json

import pytest

from anatoolbox.render import (
    CORE_RENDER_TYPES,
    MARKDOWN_RENDER_TYPE,
    build_markdown_result,
    build_render_payload,
    build_tool_result,
    register_render_type,
    render_types,
    structured_render_types,
)


@pytest.fixture(autouse=True)
def restore_render_types():
    import anatoolbox.render as r

    saved_all, saved_struct = set(r._REGISTERED), set(r._STRUCTURED)
    yield
    r._REGISTERED.clear()
    r._REGISTERED.update(saved_all)
    r._STRUCTURED.clear()
    r._STRUCTURED.update(saved_struct)


def test_core_types_are_generic_not_domain_nouns():
    assert CORE_RENDER_TYPES == {"markdown", "json", "table", "chart", "synthesis"}


def test_build_payload_is_flat():
    assert build_render_payload("json", rows=[1]) == {"render_type": "json", "rows": [1]}


def test_unknown_render_type_rejected_with_guidance():
    with pytest.raises(ValueError, match="register_render_type"):
        build_render_payload("usecases", items=[])


def test_host_can_register_its_own_type():
    register_render_type("usecases")
    assert "usecases" in render_types()
    assert "usecases" in structured_render_types()
    assert json.loads(build_tool_result(render_type="usecases", items=[])) == {
        "render": {"render_type": "usecases", "items": []}
    }


def test_register_non_structured_type():
    register_render_type("prose", structured=False)
    assert "prose" in render_types()
    assert "prose" not in structured_render_types()


def test_empty_render_type_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        register_render_type("   ")


def test_markdown_helper():
    assert json.loads(build_markdown_result("# hi")) == {
        "render": {"render_type": MARKDOWN_RENDER_TYPE, "content": "# hi"}
    }


def test_render_type_mismatch_rejected():
    payload = build_render_payload("json", rows=[])
    with pytest.raises(ValueError, match="mismatch"):
        build_tool_result(render_type="markdown", render=payload)


def test_render_types_returns_immutable_snapshot():
    assert isinstance(render_types(), frozenset)
