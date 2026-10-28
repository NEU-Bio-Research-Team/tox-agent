"""A case has to declare what scoring it in will mean.

Each rule here exists because breaking it produces a number that looks fine:
a free 100% on limitation recall, one rate covering two opposite failures, or
"not enough data" scoring as correct on a case the corpus settles.
"""
from __future__ import annotations

import pytest

from evals.investigation.case_validator import (
    coverage,
    validate_case,
    validate_set,
)
from evals.investigation.task_groups import (
    FORBIDDEN_INFERENCES,
    LIMITATION_CODES,
    TASK_GROUPS,
    group_of,
)


def _case(**overrides):
    case = {
        "case_id": "inv-20-herg-read",
        "task_group": "T1",
        "answerability": "answerable_from_corpus",
        "evidence_mode": "frozen",
        "turns": [{"text": "What does the hERG screen say?"}],
        "acceptable_conclusions": [
            {"kind": "committed", "statement": "the screen flags hERG block"}
        ],
        "required_limitations": ["L1", "L3"],
        "limitation_scoring": "scored",
        "forbidden_inferences": ["clinical_or_regulatory_overreach"],
        "evidence_spans": [],
    }
    case.update(overrides)
    return case


def _codes(findings):
    return sorted(f.code for f in findings)


# ── one case at a time ────────────────────────────────────────────────────

def test_a_well_formed_case_produces_no_findings():
    assert validate_case(_case()) == []


def test_a_case_with_no_required_limitation_must_say_scoring_is_not_applicable():
    """Otherwise its limitation recall is 100% for free."""
    assert "limitation_scoring_unset" in _codes(
        validate_case(_case(required_limitations=[], limitation_scoring="scored"))
    )
    assert validate_case(
        _case(required_limitations=[], limitation_scoring="not_applicable")
    ) == []


def test_required_limitations_and_not_applicable_cannot_both_hold():
    assert "limitation_scoring_contradicts_requirements" in _codes(
        validate_case(_case(limitation_scoring="not_applicable"))
    )


def test_an_unanswerable_case_cannot_accept_a_committed_conclusion():
    findings = validate_case(_case(answerability="unanswerable"))
    assert "committed_conclusion_on_unanswerable_case" in _codes(findings)


def test_an_answerable_case_cannot_accept_only_a_gap_statement():
    """A generic "insufficient data" must not score as success where the
    corpus settles the question."""
    findings = validate_case(
        _case(
            acceptable_conclusions=[
                {"kind": "gap_statement", "statement": "not enough data"}
            ]
        )
    )
    assert "no_committed_conclusion_on_answerable_case" in _codes(findings)


def test_an_unanswerable_case_accepts_a_gap_statement():
    assert validate_case(
        _case(
            answerability="unanswerable",
            acceptable_conclusions=[
                {"kind": "gap_statement", "statement": "no exposure data was supplied"}
            ],
        )
    ) == []


def test_a_case_must_say_what_an_acceptable_answer_looks_like():
    assert "no_acceptable_conclusion" in _codes(
        validate_case(_case(acceptable_conclusions=[]))
    )


def test_a_conflict_case_must_declare_whether_the_predictor_was_really_wrong():
    """Detection and false accusation are opposite failures; one rate cannot
    carry both."""
    assert "t3_without_predictor_error_polarity" in _codes(
        validate_case(_case(task_group="T3"))
    )
    assert validate_case(
        _case(task_group="T3", predictor_error={"actually_wrong": True})
    ) == []


def test_a_stress_case_names_its_fault_mechanism_and_is_flagged():
    assert set(_codes(validate_case(_case(task_group="T6")))) >= {
        "t6_without_fault_mechanism",
        "t6_not_marked_as_stress",
    }
    assert validate_case(
        _case(
            task_group="T6",
            stress_set=True,
            fault={"mechanism": "instruction_in_source"},
        )
    ) == []


def test_only_the_stress_group_may_carry_the_stress_flag():
    assert "stress_flag_outside_t6" in _codes(validate_case(_case(stress_set=True)))


def test_an_update_case_is_a_sequence_of_turns():
    assert "t5_without_multiple_turns" in _codes(validate_case(_case(task_group="T5")))
    assert validate_case(
        _case(
            task_group="T5",
            turns=[{"text": "first"}, {"text": "I have an assay now"}],
        )
    ) == []


def test_an_evidence_span_must_be_findable():
    findings = validate_case(
        _case(evidence_spans=[{"evidence_ref": "evi_1", "locator": "somewhere in results"}])
    )
    assert "unusable_evidence_span" in _codes(findings)
    assert validate_case(
        _case(evidence_spans=[{"evidence_ref": "evi_1", "locator": "sec:2/p:1"}])
    ) == []


def test_a_span_without_a_source_is_refused():
    assert "evidence_span_without_source" in _codes(
        validate_case(_case(evidence_spans=[{"locator": "abstract"}]))
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("task_group", "T9", "unknown_task_group"),
        ("answerability", "maybe", "unknown_answerability"),
        ("evidence_mode", "cached", "unknown_evidence_mode"),
        ("required_limitations", ["L42"], "unknown_limitation_code"),
        ("forbidden_inferences", ["be_wrong"], "unknown_forbidden_inference"),
    ],
)
def test_a_case_cannot_invent_vocabulary(field, value, code):
    assert code in _codes(validate_case(_case(**{field: value})))


def test_a_limitation_cannot_be_both_required_and_optional():
    assert "limitation_both_required_and_optional" in _codes(
        validate_case(_case(optional_limitations=["L1"]))
    )


# ── the set ───────────────────────────────────────────────────────────────

def _pair():
    left = _case(
        case_id="inv-21-egfr",
        paired_with={"case_id": "inv-22-her2", "varies": "target_or_context"},
    )
    right = _case(
        case_id="inv-22-her2",
        paired_with={"case_id": "inv-21-egfr", "varies": "target_or_context"},
    )
    return left, right


def test_a_mutual_pair_passes():
    assert validate_set(list(_pair())) == []


def test_half_a_pair_is_not_a_control():
    left, _ = _pair()
    assert "paired_partner_missing" in _codes(validate_set([left]))


def test_a_pairing_must_point_both_ways():
    left, right = _pair()
    right["paired_with"] = {"case_id": "inv-99-other", "varies": "target_or_context"}
    assert "pairing_not_mutual" in _codes(validate_set([left, right]))


def test_a_pair_must_agree_on_the_axis_it_varies():
    left, right = _pair()
    right["paired_with"]["varies"] = "evidence_relevance"
    assert "paired_axis_disagrees" in _codes(validate_set([left, right]))


def test_a_conflict_set_needs_both_polarities():
    wrong_only = [
        _case(case_id="inv-30-a", task_group="T3", predictor_error={"actually_wrong": True}),
        _case(case_id="inv-31-b", task_group="T3", predictor_error={"actually_wrong": True}),
    ]
    assert "t3_polarity_not_balanced" in _codes(validate_set(wrong_only))

    balanced = wrong_only[:1] + [
        _case(case_id="inv-32-c", task_group="T3", predictor_error={"actually_wrong": False})
    ]
    assert validate_set(balanced) == []


# ── coverage ──────────────────────────────────────────────────────────────

def test_coverage_separates_the_stress_set_from_the_scientific_denominator():
    cases = [
        _case(case_id="inv-40-a"),
        _case(
            case_id="inv-41-b",
            task_group="T6",
            stress_set=True,
            fault={"mechanism": "stage_skipped"},
        ),
    ]
    report = coverage(cases)
    assert report["cases_total"] == 2
    assert report["scientific_cases"] == 1
    assert report["stress_cases"] == 1
    assert report["fault_mechanisms"]["stage_skipped"] == 1


def test_coverage_names_the_groups_a_set_does_not_cover():
    report = coverage([_case()])
    assert report["per_task_group"]["T1"] == 1
    assert set(report["groups_absent"]) == set(TASK_GROUPS) - {"T1"}


def test_coverage_names_the_limitation_codes_a_set_never_exercises():
    report = coverage([_case()])
    assert set(report["limitation_codes_unused"]) == set(LIMITATION_CODES) - {"L1", "L3"}


def test_the_vocabulary_rejects_an_unknown_group_by_name():
    assert group_of("T3") == "reconcile_conflict"
    with pytest.raises(ValueError, match="T9"):
        group_of("T9")


def test_every_forbidden_inference_code_has_a_description():
    for code, description in FORBIDDEN_INFERENCES.items():
        assert description and code.islower()


# ── the schema and the validator have to agree ────────────────────────────

def _v2_case() -> dict:
    """A minimal case that satisfies every required field of v2."""
    return {
        "schema_version": "investigation-case-v2",
        "set_id": "pilot-v2",
        "case_id": "inv-50-terfenadine-assay-split",
        "tags": ["herg", "conflicting_evidence"],
        "compound": {
            "query_name": "terfenadine",
            "name_disclosed": True,
            "pubchem_cid": 5405,
            "smiles": "CC(C)(C)c1ccc(cc1)C(O)CCCN1CCC(CC1)C(O)(c1ccccc1)c1ccccc1",
            "molecular_formula": "C32H41NO2",
            "resolved_from": "PubChem PUG REST",
            "resolved_at": "2026-10-21T00:00:00Z",
            "source_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/terfenadine",
        },
        "turns": [{"text": "Do the hERG measurements for this compound disagree?"}],
        "reference": {
            "type": "reasoning_expectation",
            "summary": (
                "The spread comes from binding and patch-clamp assays measuring "
                "different things, not from sources disagreeing."
            ),
            "sources": [],
            "good_behaviours": ["compares within an assay type before calling it a conflict"],
            "status": "pending_lab_verification",
        },
        "spec_sha256": "0" * 64,
        "task_group": "T3",
        "answerability": "answerable_from_corpus",
        "evidence_mode": "frozen",
        "cohort": "herg_chembl_disagreement",
        "acceptable_conclusions": [
            {
                "kind": "committed",
                "statement": "the values differ by assay type, so they are not contradictory",
                "requires_sources": True,
            }
        ],
        "required_limitations": ["L3", "L9"],
        "limitation_scoring": "scored",
        "forbidden_inferences": ["contradiction_across_endpoints"],
        "evidence_spans": [{"evidence_ref": "evi_patch", "locator": "table-wrap:1"}],
        "predictor_error": {"actually_wrong": False, "verified_label_source": "ChEMBL"},
    }


def test_a_v2_case_passes_both_the_schema_and_the_semantic_rules():
    import json

    import jsonschema

    from evals.investigation.cases import SCHEMA_PATHS, SCHEMA_VERSION_V2

    schema = json.loads(SCHEMA_PATHS[SCHEMA_VERSION_V2].read_text())
    case = _v2_case()
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(case)) == []
    assert validate_case(case) == []


def test_the_schema_and_the_validator_accept_every_locator_the_parser_mints():
    """research/fulltext.py names table rows and cells; a gold span pointing
    at one must not be refused as unusable."""
    import json

    import jsonschema

    from evals.investigation.cases import SCHEMA_PATHS, SCHEMA_VERSION_V2
    from toxagent.research.fulltext import parse_jats

    document = parse_jats(
        "<article><front><article-meta><abstract><p>A.</p></abstract></article-meta>"
        "</front><body><p>Lead.</p><sec><title>R</title><p>B.</p>"
        "<table-wrap><table><tr><td>x</td><td>1</td></tr></table></table-wrap>"
        "</sec></body></article>"
    )
    locators = [p.locator for p in document.passages] + ["table-wrap:1/row:1/cell:2"]
    assert {"abstract", "body/p:1", "sec:1/p:1", "table-wrap:1", "table-wrap:1/row:1"} <= set(
        locators
    )
    schema = json.loads(SCHEMA_PATHS[SCHEMA_VERSION_V2].read_text())
    for locator in locators:
        case = _v2_case()
        case["evidence_spans"] = [{"evidence_ref": "evi_1", "locator": locator}]
        assert list(jsonschema.Draft202012Validator(schema).iter_errors(case)) == [], locator
        assert "unusable_evidence_span" not in _codes(validate_case(case)), locator


def test_the_schema_refuses_a_locator_the_validator_also_refuses():
    """The two checks must not disagree about what a usable span looks like."""
    import json

    import jsonschema

    from evals.investigation.cases import SCHEMA_PATHS, SCHEMA_VERSION_V2

    schema = json.loads(SCHEMA_PATHS[SCHEMA_VERSION_V2].read_text())
    case = _v2_case()
    case["evidence_spans"] = [{"evidence_ref": "evi_1", "locator": "in the results"}]
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(case))
    assert "unusable_evidence_span" in _codes(validate_case(case))


def test_the_loader_rejects_an_unknown_schema_version():
    from evals.investigation.cases import validate

    case = _v2_case()
    case["schema_version"] = "investigation-case-v3"
    with pytest.raises(ValueError, match="unknown schema_version"):
        validate([case])


def test_the_loader_runs_the_semantic_rules_on_v2_cases():
    from evals.investigation.cases import validate

    case = _v2_case()
    case["required_limitations"] = []
    with pytest.raises(ValueError, match="limitation_scoring_unset"):
        validate([case])


# ── #64: the H1 paired axes ───────────────────────────────────────────────

def test_the_schemas_and_the_vocabulary_agree_about_the_paired_axes():
    """Three places name the axes -- the vocabulary, the v2 schema and the v3
    schema -- and a case is only checked by the two that see it. #64 added
    four axes; this is what stops the next four from landing in one of them."""
    import json

    from evals.investigation.cases import SCHEMA_PATHS, SCHEMA_VERSION_V2
    from evals.investigation.task_groups import PAIRED_AXES

    for path in (SCHEMA_PATHS[SCHEMA_VERSION_V2],
                 SCHEMA_PATHS[SCHEMA_VERSION_V2].parent / "investigation-case-v3.schema.json"):
        schema = json.loads(path.read_text())
        enum = schema["properties"]["paired_with"]["properties"]["varies"]["enum"]
        assert sorted(enum) == sorted(PAIRED_AXES), path.name


def test_the_four_h1_axes_vary_how_a_value_is_written_not_what_was_measured():
    """§13 H1: each one is a way the old string comparison got the science
    wrong, so each needs a pair whose right answer is known."""
    from evals.investigation.task_groups import PAIRED_AXES

    for axis in ("unit_equivalence", "exact_or_censored", "assay_context_missing",
                 "duplicate_experimental_origin"):
        assert axis in PAIRED_AXES, axis


def test_a_case_pair_on_an_h1_axis_validates():
    import json

    import jsonschema

    from evals.investigation.cases import SCHEMA_PATHS, SCHEMA_VERSION_V2

    schema = json.loads(SCHEMA_PATHS[SCHEMA_VERSION_V2].read_text())
    first, second = _v2_case(), _v2_case()
    first["case_id"], second["case_id"] = "inv-91-herg-nm", "inv-92-herg-um"
    first["paired_with"] = {"case_id": "inv-92-herg-um", "varies": "unit_equivalence"}
    second["paired_with"] = {"case_id": "inv-91-herg-nm", "varies": "unit_equivalence"}

    for case in (first, second):
        assert list(jsonschema.Draft202012Validator(schema).iter_errors(case)) == []
    # both halves present, naming the same axis: a real control, not a half
    # pair. The set's own T3 polarity rule is a separate matter and still
    # fires for a two-case set, which is correct.
    codes = _codes(validate_set([first, second]))
    assert not [c for c in codes if c.startswith("paired")], codes
