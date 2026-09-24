"""The audit captures parse, name their finding, and carry no secrets.

A fixture that quietly stopped reproducing its finding is worse than no
fixture: the test that depends on it keeps passing for the wrong reason.
"""
from __future__ import annotations

import re

import pytest

from tests.support.audit_fixtures import (
    ALL_FIXTURES,
    BASELINE_MANIFEST,
    CCO_ATTRIBUTION,
    DUPLICATE_USAGE_SSE,
    ETHANOL_HERG_FALSE_MATCHES,
    FIXTURE_ROOT,
    REPORT_CONTRADICTION,
    load,
)

#: Anything shaped like a credential. The audit ran against a live stack with a
#: real provider key and a run-scoped bearer token; neither may be in the tree.
_SECRET_SHAPES = (
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9]{8,}"),
    re.compile(r"\bAuthorization\b", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\b", re.IGNORECASE),
    re.compile(r"\bey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."),  # a JWT
)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_every_fixture_parses(name: str) -> None:
    assert isinstance(load(name), dict)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_no_fixture_carries_a_credential(name: str) -> None:
    text = (FIXTURE_ROOT / name).read_text(encoding="utf-8")
    for pattern in _SECRET_SHAPES:
        assert not pattern.search(text), f"{name} matches {pattern.pattern}"


@pytest.mark.parametrize(
    "name,finding",
    [
        (DUPLICATE_USAGE_SSE, "P1-1"),
        (ETHANOL_HERG_FALSE_MATCHES, "P1-2"),
        (REPORT_CONTRADICTION, "P0-2"),
        (CCO_ATTRIBUTION, "P1-6"),
    ],
)
def test_fixture_names_the_finding_it_reproduces(name: str, finding: str) -> None:
    assert load(name)["finding"] == finding


def test_duplicate_usage_fixture_actually_repeats_a_snapshot() -> None:
    fixture = load(DUPLICATE_USAGE_SSE)
    snapshots = [
        (props.get("part") or props.get("info") or {}).get("tokens")
        for props in (envelope["properties"] for envelope in fixture["envelopes"])
    ]
    repeated = [s for s in snapshots if s and s.get("input") == 3826]
    assert len(repeated) == 3, "the fixture must still show the same snapshot three times"
    assert fixture["expected_normalized"]["facts"] == 2


def test_evidence_fixture_holds_five_false_matches_and_a_control() -> None:
    fixture = load(ETHANOL_HERG_FALSE_MATCHES)
    assert len(fixture["hits"]) == 5
    assert all(
        hit["expected_assessment"]["relevance"] == "irrelevant" for hit in fixture["hits"]
    )
    assert fixture["expected"]["promoted"] == 0
    assert {hit["expected_assessment"]["relevance"] for hit in fixture["control_hits"]} == {
        "direct",
        "contextual",
    }


def test_report_fixture_still_contradicts_itself() -> None:
    fixture = load(REPORT_CONTRADICTION)
    summary = next(
        section for section in fixture["sections"] if section["section_id"] == "executive_summary"
    )
    assert "no unmapped mass" in summary["body_markdown"]
    highlights = fixture["context"]["explanations"][0]["highlights"]
    assert len(highlights["negative_contributors"]) == 3
    assert highlights["unmapped_importance"] > 0
    assert fixture["context"]["include_external_evidence"] is False
    assert any(item["code"] == "evidence_scope_limited" for item in fixture["limitations"])


def test_attribution_fixture_mass_adds_up() -> None:
    fixture = load(CCO_ATTRIBUTION)
    payload = fixture["payload"]
    expected = fixture["expected_coverage"]
    mapped = sum(atom["importance"] for atom in payload["atoms"])
    total = mapped + sum(
        token["importance"] for token in payload["tokens"] if token["token"].startswith("[")
    )
    assert total == pytest.approx(1.0, abs=1e-9)
    assert mapped == pytest.approx(expected["mapped_importance_fraction"], abs=1e-9)
    assert payload["unmapped_importance"] == pytest.approx(
        expected["special_token_importance_fraction"], abs=1e-12
    )


def test_baseline_manifest_states_a_gate_for_every_kpi() -> None:
    manifest = load(BASELINE_MANIFEST)
    assert manifest["environment"]["opencode_version"] == "1.17.11"
    for name, row in manifest["kpis"].items():
        assert any(key.startswith("gate") for key in row), f"{name} has no gate"
