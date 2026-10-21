"""Whether a record is about a screening's ligand on its target (AG-EVIDENCE).

The same conservative rules as the endpoint check, over a target from the
versioned catalog: the ligand on the target is direct; the ligand on another
target is contextual, never direct; the target with another ligand is not
about the subject; a target the catalog does not hold leaves even a ligand
match uncertain; and thin metadata is never a match by default.
"""
from __future__ import annotations

from types import SimpleNamespace

from toxagent.research import target_catalog
from toxagent.research.relevance import (
    TARGET_RELEVANCE_POLICY_VERSION,
    CompoundIdentity,
    Relevance,
    RelevanceTarget,
    assess,
)

LIGAND = CompoundIdentity(canonical_smiles="CCO", preferred_name="zaltinib")
EGFR = target_catalog.lookup("EGFR")


def _hit(title: str, abstract: str = "") -> SimpleNamespace:
    return SimpleNamespace(title=title, abstract_or_excerpt=abstract, normalized_facts={},
                           identifier=None)


def _target() -> RelevanceTarget:
    return RelevanceTarget(target_id=EGFR.target_id, target_terms=EGFR.vocabulary)


def test_the_ligand_on_the_target_is_direct_and_states_what_it_established():
    result = assess(_hit("Zaltinib inhibits EGFR kinase with an IC50 of 12 nM"),
                    compound=LIGAND, target=_target())
    assert result.relevance is Relevance.DIRECT
    assert result.reason_codes == ("compound_match", "target_match")
    assert result.assessed_scope == {"ligand_id": "CCO", "target_id": "EGFR"}
    assert result.policy_version == TARGET_RELEVANCE_POLICY_VERSION


def test_the_ligand_on_another_target_is_contextual_never_direct():
    result = assess(_hit("Zaltinib blocks the hERG potassium channel"),
                    compound=LIGAND, target=_target())
    assert result.relevance is Relevance.CONTEXTUAL
    assert "target_mismatch" in result.reason_codes
    assert "target_id" not in result.assessed_scope


def test_the_target_with_another_ligand_is_not_about_the_subject():
    result = assess(_hit("Gefitinib inhibits EGFR"), compound=LIGAND, target=_target())
    assert result.relevance is Relevance.IRRELEVANT
    assert result.reason_codes == ("compound_mismatch", "target_match")


def test_a_target_outside_the_catalog_is_never_a_match_by_default():
    result = assess(_hit("Zaltinib inhibits BRAF"), compound=LIGAND,
                    target=RelevanceTarget(target_id="BRAF", uncataloged=True))
    assert result.relevance is Relevance.UNCERTAIN and not result.is_citable
    assert "target_not_in_catalog" in result.reason_codes


def test_thin_metadata_is_uncertain():
    assert assess(_hit(""), compound=LIGAND, target=_target()).relevance is Relevance.UNCERTAIN


def test_the_catalog_refuses_a_target_whose_uniprot_disagrees():
    assert target_catalog.lookup("EGFR", uniprot_id="P00533") is EGFR
    assert target_catalog.lookup("EGFR", uniprot_id="Q12809") is None
    assert target_catalog.lookup("KCNH2").toxicity_endpoint == "herg"
    assert target_catalog.lookup("NOPE") is None


def test_the_endpoint_check_is_unchanged():
    result = assess(_hit("Zaltinib and hERG blockade"), compound=LIGAND,
                    target=RelevanceTarget(endpoint="herg"))
    assert result.relevance is Relevance.DIRECT
    assert result.reason_codes == ("compound_match", "endpoint_match")
    assert result.assessed_scope == {}
