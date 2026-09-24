"""Projections keep endpoint semantics (plan sections 2.2, 8.4)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.application import projections
from toxagent.domain.analysis import AnalysisSnapshot, PredictorProvenance
from toxagent.domain.errors import InvalidRequest
from toxagent.domain.ids import new_id
from tests.support.predictor import ASPIRIN, prediction

NOW = datetime(2026, 9, 4, tzinfo=timezone.utc)


def snapshot(endpoints=("herg", "tox21"), requested=None) -> AnalysisSnapshot:
    return AnalysisSnapshot.create(
        session_id=new_id("ses"), run_id=new_id("run"), input_smiles=ASPIRIN,
        requested_endpoints=tuple(requested or endpoints),
        predictor_response=prediction(endpoints=endpoints),
        provenance=PredictorProvenance(base_url_id="toxpred-local", artifact_hashes=("a",)),
        policy_snapshot={}, now=NOW,
    )


def test_the_display_projection_has_no_aggregate_anything():
    display = projections.display_projection(snapshot())
    flat = str(display).lower()
    for forbidden in ("overall", "aggregate", "severity", "total_risk", "safety_score"):
        assert forbidden not in flat


def test_tox21_stays_twelve_measurements_not_a_count():
    display = projections.display_projection(snapshot())
    tox21 = display["sections"]["tox21"]
    assert len(tox21["assays"]) == 12
    assert "hits" not in tox21 and "active_count" not in tox21


def test_unavailable_endpoints_are_named_in_the_projection():
    display = projections.display_projection(
        snapshot(endpoints=("herg",), requested=("herg", "clintox"))
    )
    assert display["unavailable_endpoints"] == ["clintox"]
    assert "endpoint_unavailable" in display["required_limitations"]


def test_the_model_projection_lists_sections_without_handing_over_values():
    view = projections.model_projection(snapshot())
    assert view["available_sections"] == ["herg", "tox21", "applicability", "provenance"]
    assert "0.73064" not in str(view)
    assert "uncalibrated_probability" in view["required_limitations"]


def test_a_slice_returns_the_field_path_of_every_value():
    result = projections.slice_analysis(snapshot(), "herg", ["probability_blocker", "label"])
    assert result["values"]["probability_blocker"] == {
        "value": 0.73064, "field_path": "predictions.herg.probability_blocker"
    }
    assert result["values"]["label"]["field_path"] == "predictions.herg.label"


def test_a_slice_cannot_reach_a_field_the_product_did_not_declare():
    with pytest.raises(InvalidRequest, match="not exposed"):
        projections.slice_analysis(snapshot(), "herg", ["probability_clinical_toxicity"])


def test_a_slice_of_an_unserved_endpoint_is_refused_rather_than_substituted():
    with pytest.raises(InvalidRequest, match="no clintox section"):
        projections.slice_analysis(
            snapshot(endpoints=("herg",), requested=("herg", "clintox")), "clintox"
        )


def test_a_tox21_slice_addresses_one_assay():
    result = projections.slice_analysis(snapshot(), "tox21", task="SR-MMP")
    assert result["values"]["active"]["field_path"] == "predictions.tox21.assays.SR-MMP.active"
    assert "not a severity score" in result["note"]


def test_an_unknown_assay_is_refused():
    with pytest.raises(InvalidRequest, match="unknown Tox21 assay"):
        projections.slice_analysis(snapshot(), "tox21", task="SR-NOPE")


def test_every_slice_carries_its_required_limitations():
    for section in ("herg", "tox21", "applicability"):
        result = projections.slice_analysis(snapshot(), section)
        assert "uncalibrated_probability" in result["required_limitations"]
        assert "applicability_is_rule_based" in result["required_limitations"]
