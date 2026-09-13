"""The prefix contracts are the package's public surface. Pin their shape.

These files are generated from the workflow spec, so a regeneration that
drops a field or changes a quote style must fail here rather than silently
disappear from prefix discovery (which is exactly what happened once).
"""

import importlib
import pathlib
import re

import pytest

import anatoolbox
from anatoolbox.stages import STAGE_ORDER, STAGE_PACKAGE_BY_LABEL, prefix_to_stage

ROOT = pathlib.Path(anatoolbox.__file__).parent
CONTRACTS = sorted(ROOT.glob("*/*/base.py"))


def test_every_stage_has_at_least_one_prefix():
    by_stage = prefix_to_stage()
    for stage in STAGE_ORDER:
        assert any(s == stage for s in by_stage.values()), f"{stage} has no prefixes"


def test_expected_contract_count():
    assert len(CONTRACTS) == 24
    assert len(prefix_to_stage()) == 24, "a contract is missing from prefix discovery"


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_contract_declares_full_metadata(path):
    """Assert on imported attributes, not source text.

    Contracts mix generated single-line assignments with hand-written
    parenthesized ones; matching the source only tests the formatting.
    """
    rel = path.parent.relative_to(ROOT).as_posix().replace("/", ".")
    mod = importlib.import_module(f"anatoolbox.{rel}.base")
    for field in (
        "PREFIX",
        "STAGE",
        "STAGE_LABEL",
        "ABSTRACT_INPUT",
        "ABSTRACT_OUTPUT",
        "TRANSFORMATION",
    ):
        value = getattr(mod, field, None)
        assert isinstance(value, str) and value.strip(), f"{rel}: {field} missing or empty"
    assert re.search(r"^class \w+Tool\(Protocol\)", path.read_text(), re.M), f"{path}: no Protocol"


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_contract_is_importable_and_consistent(path):
    rel = path.parent.relative_to(ROOT).as_posix().replace("/", ".")
    mod = importlib.import_module(f"anatoolbox.{rel}")
    assert mod.PREFIX.endswith("_"), f"{rel}: prefix must end with '_'"
    assert mod.STAGE in STAGE_ORDER
    # the directory name must match the prefix it declares
    assert path.parent.name == mod.PREFIX.rstrip("_")
    assert mod.TRANSFORMATION.strip(), f"{rel}: empty transformation"


def test_prefixes_are_unique():
    prefixes = [p.parent.name for p in CONTRACTS]
    assert len(prefixes) == len(set(prefixes))


def test_stage_packages_match_stage_order():
    """No orphaned stage directory survives a restructure, such as a removed `present/`."""
    assert {path.parent.parent.name for path in CONTRACTS} == set(STAGE_ORDER)


@pytest.mark.parametrize("path", CONTRACTS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_contract_sits_in_the_stage_it_declares(path):
    """A contract moved between stages must have its STAGE and STAGE_LABEL moved too."""
    rel = path.parent.relative_to(ROOT).as_posix().replace("/", ".")
    mod = importlib.import_module(f"anatoolbox.{rel}.base")
    assert path.parent.parent.name == mod.STAGE
    assert STAGE_PACKAGE_BY_LABEL[mod.STAGE_LABEL] == mod.STAGE
