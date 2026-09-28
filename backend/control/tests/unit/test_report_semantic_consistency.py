"""P0-2: the audit's report must not be publishable.

`rpt_d595fee878984745b916c1c396ddbcde` passed every gate the validator had and
shipped with an executive summary denying the explanation data in its own
artifact, plus a limitation describing a literature search the build never ran.

The golden case is the whole point of this file: the sanitized artifact goes in
and violations come out. The rest pins the boundaries, because a gate that
fires on a correct sentence is a gate someone will turn off.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.support.audit_fixtures import REPORT_CONTRADICTION, load
from toxagent.domain.xai_coverage import compute_coverage
from toxagent.domain.report import (
    ExplanationHighlights,
    ExplanationPackage,
    ExplanationStatus,
)
from toxagent.validation.report_semantics import (
    EVIDENCE_INSUFFICIENT,
    EVIDENCE_NOT_REQUESTED,
    EVIDENCE_PROVIDER_FAILED,
    EVIDENCE_ZERO_RESULTS,
    EvidenceSituation,
    canonical_explanation_line,
    check_evidence_scope_consistency,
    check_evidence_state_gap,
    check_explanation_consistency,
)

FIXTURE = load(REPORT_CONTRADICTION)


def _section(section_id: str, body: str) -> SimpleNamespace:
    return SimpleNamespace(section_id=section_id, body_markdown=body)


def _fixture_sections() -> list[SimpleNamespace]:
    return [_section(s["section_id"], s["body_markdown"]) for s in FIXTURE["sections"]]


def _fixture_limitations() -> list[SimpleNamespace]:
    return [SimpleNamespace(code=l["code"], text=l["text"]) for l in FIXTURE["limitations"]]


def _fixture_gaps() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            gap_id=g["gap_id"], reason=g["reason"], section_id=g["section_id"]
        )
        for g in FIXTURE["gaps"]
    ]


def _package(*, positive: int = 0, negative: int = 3, unmapped: float | None = 0.3584):
    return ExplanationPackage(
        explanation_id="xpl_" + "0" * 32,
        observation_id="obs_" + "0" * 32,
        endpoint="herg",
        task=None,
        method="integrated-gradients",
        status=ExplanationStatus.COMPLETED,
        highlights=ExplanationHighlights(
            positive_contributors=tuple({"atom_index": i} for i in range(positive)),
            negative_contributors=tuple({"atom_index": i} for i in range(negative)),
            unmapped_importance=unmapped,
        ),
    )


def _fixture_package() -> ExplanationPackage:
    data = FIXTURE["context"]["explanations"][0]
    highlights = data["highlights"]
    return ExplanationPackage(
        explanation_id=data["explanation_id"],
        observation_id=data["observation_id"],
        endpoint=data["endpoint"],
        task=data["task"],
        method=data["method"],
        status=ExplanationStatus(data["status"]),
        highlights=ExplanationHighlights(
            positive_contributors=tuple(highlights["positive_contributors"]),
            negative_contributors=tuple(highlights["negative_contributors"]),
            unmapped_importance=highlights["unmapped_importance"],
        ),
    )


# --- the golden case --------------------------------------------------------


def test_the_audit_artifact_is_refused() -> None:
    package = _fixture_package()
    situation = EvidenceSituation(
        requested=FIXTURE["context"]["include_external_evidence"], search_performed=False
    )
    violations = [
        *check_explanation_consistency(
            sections=_fixture_sections(), explanations={package.explanation_id: package}
        ),
        *check_evidence_scope_consistency(
            sections=_fixture_sections(),
            limitations=_fixture_limitations(),
            situation=situation,
        ),
        *check_evidence_state_gap(gaps=_fixture_gaps(), situation=situation),
    ]
    codes = {violation.code for violation in violations}
    for expected in FIXTURE["expected_violations"]:
        assert expected in codes, f"{expected} was not raised; got {sorted(codes)}"


def test_the_executive_summary_is_the_section_named() -> None:
    package = _fixture_package()
    violations = check_explanation_consistency(
        sections=_fixture_sections(), explanations={package.explanation_id: package}
    )
    assert violations
    assert all("executive_summary" in v.path for v in violations)
    assert any("contributor" in v.message for v in violations)
    assert any("unmapped" in v.message for v in violations)


def test_the_evidence_state_gap_is_mislabelled_in_the_audit_artifact() -> None:
    """The build never searched, and the gap called it 'no relevant evidence'
    — which tells a reader the literature is thin rather than unexamined."""
    situation = EvidenceSituation(requested=False, search_performed=False)
    assert situation.state == EVIDENCE_NOT_REQUESTED
    violations = check_evidence_state_gap(gaps=_fixture_gaps(), situation=situation)
    assert [v.code for v in violations] == ["evidence_state_mislabelled"]
    assert violations[0].expected == "external_evidence_not_requested"


# --- the boundaries ---------------------------------------------------------


def test_a_true_denial_is_not_a_violation() -> None:
    """An explanation that genuinely has no contributors may say so."""
    package = _package(positive=0, negative=0, unmapped=0.0)
    sections = [
        _section(
            "executive_summary",
            "The attribution reports no contributors and no unmapped mass.",
        )
    ]
    assert check_explanation_consistency(
        sections=sections, explanations={package.explanation_id: package}
    ) == []


def test_a_correct_description_is_not_a_violation() -> None:
    package = _package()
    sections = [
        _section(
            "explanation_and_visuals",
            "Three atoms contribute negatively, and 35.8% of the attribution mass "
            "is unmapped.",
        )
    ]
    assert check_explanation_consistency(
        sections=sections, explanations={package.explanation_id: package}
    ) == []


def test_one_endpoint_without_contributors_may_be_described_as_such() -> None:
    """A multi-endpoint report where one explanation genuinely has none must
    still be able to say so without tripping the gate."""
    with_contributors = _package()
    without = ExplanationPackage(
        explanation_id="xpl_" + "1" * 32,
        observation_id="obs_" + "1" * 32,
        endpoint="clintox",
        task=None,
        method="integrated-gradients",
        status=ExplanationStatus.COMPLETED,
        highlights=ExplanationHighlights(unmapped_importance=0.0),
    )
    sections = [
        _section("explanation_and_visuals", "The ClinTox attribution has no contributors.")
    ]
    assert check_explanation_consistency(
        sections=sections,
        explanations={
            with_contributors.explanation_id: with_contributors,
            without.explanation_id: without,
        },
    ) == []


def test_an_unreported_unmapped_mass_does_not_trip_the_unmapped_gate() -> None:
    """The explainer did not say. A gate that fired here would be asserting a
    fact the server does not hold, which is the mistake it exists to catch."""
    package = _package(unmapped=None)
    sections = [_section("executive_summary", "There is no unmapped mass.")]
    assert check_explanation_consistency(
        sections=sections, explanations={package.explanation_id: package}
    ) == []


def test_a_build_with_no_explanations_is_not_checked() -> None:
    assert check_explanation_consistency(
        sections=[_section("executive_summary", "no contributors at all")], explanations={}
    ) == []


@pytest.mark.parametrize(
    "body",
    [
        "Phân tích attribution không có nguyên tử đóng góp nào.",
        "Kết quả không có khối lượng chưa ánh xạ.",
    ],
)
def test_the_denial_is_caught_in_vietnamese_too(body: str) -> None:
    package = _package()
    violations = check_explanation_consistency(
        sections=[_section("executive_summary", body)],
        explanations={package.explanation_id: package},
    )
    assert violations


# --- evidence scope ---------------------------------------------------------


def test_a_limitation_describing_an_unperformed_search_is_refused() -> None:
    situation = EvidenceSituation(requested=False, search_performed=False)
    violations = check_evidence_scope_consistency(
        sections=[],
        limitations=[
            SimpleNamespace(
                code="evidence_scope_limited",
                text="The literature search covered a single provider and did not read "
                "full texts.",
            )
        ],
        situation=situation,
    )
    assert [v.code for v in violations] == ["evidence_scope_limitation_without_search"]


def test_the_same_limitation_code_is_fine_when_worded_honestly() -> None:
    """The code is not the problem — a report with no external evidence does
    have a limited evidence scope. The wording was."""
    situation = EvidenceSituation(requested=False, search_performed=False)
    assert check_evidence_scope_consistency(
        sections=[],
        limitations=[
            SimpleNamespace(
                code="evidence_scope_limited",
                text="No external literature was consulted for this build, so nothing "
                "here is corroborated against published data.",
            )
        ],
        situation=situation,
    ) == []


def test_a_build_that_did_search_may_describe_its_search() -> None:
    situation = EvidenceSituation(
        requested=True, search_performed=True, candidates_found=12, promoted=2
    )
    assert situation.state == "cited"
    assert check_evidence_scope_consistency(
        sections=[
            _section("external_evidence", "The search covered EuropePMC abstracts only.")
        ],
        limitations=[],
        situation=situation,
    ) == []


def test_promoted_records_are_proof_a_search_happened() -> None:
    """A caller that has not yet been taught to pass search_performed must not
    make a citing report look like one that never looked."""
    situation = EvidenceSituation(requested=True, search_performed=False, promoted=3)
    assert situation.searched is True
    assert situation.state == "cited"


# --- the four states --------------------------------------------------------


@pytest.mark.parametrize(
    "situation,state,reason",
    [
        (
            EvidenceSituation(requested=False, search_performed=False),
            EVIDENCE_NOT_REQUESTED,
            "external_evidence_not_requested",
        ),
        (
            EvidenceSituation(requested=True, search_performed=True, candidates_found=0),
            EVIDENCE_ZERO_RESULTS,
            "no_relevant_evidence",
        ),
        (
            EvidenceSituation(requested=True, search_performed=True, provider_failed=True),
            EVIDENCE_PROVIDER_FAILED,
            "provider_unavailable",
        ),
        (
            EvidenceSituation(
                requested=True, search_performed=True, candidates_found=9, promoted=0
            ),
            EVIDENCE_INSUFFICIENT,
            "no_relevant_evidence",
        ),
    ],
)
def test_the_four_states_are_distinct_and_each_names_its_gap(
    situation, state, reason
) -> None:
    assert situation.state == state
    gaps = [
        SimpleNamespace(gap_id="g1", reason=reason, section_id="external_evidence")
    ]
    assert check_evidence_state_gap(gaps=gaps, situation=situation) == []


def test_a_missing_evidence_gap_is_itself_a_violation() -> None:
    situation = EvidenceSituation(requested=True, search_performed=True, candidates_found=0)
    violations = check_evidence_state_gap(gaps=[], situation=situation)
    assert [v.code for v in violations] == ["evidence_state_not_disclosed"]


def test_a_citing_build_needs_no_evidence_gap() -> None:
    situation = EvidenceSituation(
        requested=True, search_performed=True, candidates_found=4, promoted=1
    )
    assert check_evidence_state_gap(gaps=[], situation=situation) == []


# --- one source of wording --------------------------------------------------


def test_the_canonical_line_is_what_sections_must_reuse() -> None:
    coverage = compute_coverage(
        {"unmapped_importance": 0.3584, "tokens": [{"token": "[CLS]", "importance": 0.3584},
                                                   {"token": "C", "importance": 0.6416}]},
        negative_contributor_count=3,
    )
    line = canonical_explanation_line(coverage)
    assert line == coverage.summary_sentence()
    assert "64.2%" in line
    # And the line the compiler produces must not itself trip the gate.
    package = _package()
    assert check_explanation_consistency(
        sections=[_section("executive_summary", line)],
        explanations={package.explanation_id: package},
    ) == []
