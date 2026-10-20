"""Whether the text was read, and what the record's own facts establish.

Three facts that had been one: no evidence found, we only saw the abstract,
and the text exists and we could not open it.
"""
from __future__ import annotations

import pytest

from toxagent.domain.evidence_relation import EvidenceScope, RelationLabel
from toxagent.research.fulltext import MAX_PASSAGE_CHARS, parse_jats
from toxagent.research.measurement import (
    measured_value,
    measurement_scope,
    normalised_record,
)
from toxagent.research.providers.europepmc import hit_from_record
from toxagent.research.text_access import (
    ACCESS_KEY,
    FAILURE_KEY,
    SIGNALS_KEY,
    TextAccess,
    access_facts,
    access_of,
    after_failed_fetch,
    after_successful_fetch,
    full_text_was_read,
)


# ── access state ──────────────────────────────────────────────────────────

def test_an_abstract_with_no_full_text_is_abstract_only():
    facts = access_facts({"inEPMC": "N", "isOpenAccess": "N"}, has_abstract=True)
    assert facts[ACCESS_KEY] == TextAccess.ABSTRACT_ONLY.value


def test_a_record_in_the_providers_full_text_store_is_available_not_retrieved():
    """Declared available is not read. Nothing has fetched it yet."""
    facts = access_facts({"inEPMC": "Y", "isOpenAccess": "Y"}, has_abstract=True)
    assert facts[ACCESS_KEY] == TextAccess.FULL_TEXT_AVAILABLE.value
    assert not full_text_was_read(facts)


def test_open_access_alone_is_a_licence_not_a_location():
    """isOpenAccess says the paper may be read somewhere, not that this
    provider can serve it."""
    facts = access_facts({"isOpenAccess": "Y", "inEPMC": "N"}, has_abstract=True)
    assert facts[ACCESS_KEY] == TextAccess.ABSTRACT_ONLY.value
    assert facts[SIGNALS_KEY]["isOpenAccess"] is True


def test_an_absent_signal_is_unknown_and_not_a_denial():
    facts = access_facts({}, has_abstract=True)
    assert facts[SIGNALS_KEY]["inEPMC"] is None
    assert facts[ACCESS_KEY] == TextAccess.ABSTRACT_ONLY.value


def test_metadata_with_no_abstract_and_no_signals_is_not_assessed():
    """Not an absence of evidence, and not an abstract either."""
    facts = access_facts({}, has_abstract=False)
    assert facts[ACCESS_KEY] == TextAccess.NOT_ASSESSED.value


def test_a_fetch_that_failed_is_recorded_with_its_reason():
    facts = access_facts({"inEPMC": "Y"}, has_abstract=True)
    failed = after_failed_fetch(facts, reason="the provider holds no full text")
    assert failed[ACCESS_KEY] == TextAccess.ACCESS_FAILED.value
    assert failed[FAILURE_KEY]
    # and it is no longer claiming the text is available
    assert access_of(failed) is TextAccess.ACCESS_FAILED


def test_a_successful_fetch_records_how_many_passages_were_held():
    facts = after_successful_fetch(
        access_facts({"inEPMC": "Y"}, has_abstract=True), passage_count=14
    )
    assert full_text_was_read(facts)
    assert facts["full_text_passage_count"] == 14
    assert FAILURE_KEY not in facts


def test_a_record_written_before_this_existed_reads_as_not_assessed():
    assert access_of({"journal": "Br J Pharmacol"}) is TextAccess.NOT_ASSESSED


def test_a_search_hit_carries_its_access_state():
    hit = hit_from_record(
        {
            "id": "12345",
            "source": "MED",
            "title": "hERG block by astemizole",
            "abstractText": "We measured IC50 values.",
            "inEPMC": "Y",
            "isOpenAccess": "Y",
            "fullTextUrlList": {
                "fullTextUrl": [
                    {"availability": "Open access", "url": "https://example.org/a"}
                ]
            },
        }
    )
    assert hit is not None
    assert hit.normalized_facts[ACCESS_KEY] == TextAccess.FULL_TEXT_AVAILABLE.value
    assert hit.normalized_facts["full_text_url"] == "https://example.org/a"


# ── located passages ──────────────────────────────────────────────────────

_JATS = """<article>
  <front><article-meta>
    <abstract><p>Astemizole blocks hERG with an IC50 of 1 nM.</p></abstract>
  </article-meta></front>
  <body>
    <sec><title>Methods</title>
      <p>Whole-cell patch clamp at 22 degrees.</p>
      <p>Cells were HEK293.</p>
    </sec>
    <sec><title>Results</title>
      <p>The measured IC50 was 0.9 nM.</p>
      <table-wrap>
        <caption><p>Table 1. Potencies</p></caption>
        <table><tr><td>Astemizole</td><td>0.9 nM</td></tr></table>
      </table-wrap>
    </sec>
  </body>
</article>"""


def test_every_passage_knows_where_it_came_from():
    document = parse_jats(_JATS)
    locators = [p.locator for p in document.passages]
    assert "abstract" in locators
    assert "sec:2/p:1" in locators
    assert "table-wrap:1" in locators
    assert document.section_titles == ("Methods", "Results")

    results = document.passage("sec:2/p:1")
    assert results is not None
    assert results.section == "Results"
    assert "0.9 nM" in results.text


def test_a_table_is_kept_as_its_own_passage_with_its_caption():
    table = parse_jats(_JATS).passage("table-wrap:1")
    assert table is not None
    assert table.kind == "table"
    assert "Table 1" in table.section
    assert "Astemizole" in table.text


def test_a_long_passage_is_clipped_rather_than_dropped():
    long_text = "value " * 1000
    document = parse_jats(
        f"<article><body><sec><title>R</title><p>{long_text}</p></sec></body></article>"
    )
    passage = document.passage("sec:1/p:1")
    assert passage is not None
    assert passage.truncated
    assert len(passage.text) <= MAX_PASSAGE_CHARS + 2


def test_malformed_xml_yields_no_passages_rather_than_raising():
    assert parse_jats("<article><body><sec>").passages == ()


# ── the normalised record ─────────────────────────────────────────────────

_CHEMBL_FACTS = {
    "molecule_chembl_id": "CHEMBL1220",
    "target_chembl_id": "CHEMBL240",
    "standard_type": "IC50",
    "standard_relation": "=",
    "standard_value": "0.9",
    "standard_units": "nM",
    "assay_type": "F",
    "assay_description": "Whole-cell patch clamp, HEK293",
}


def test_a_record_projects_into_the_one_scope_vocabulary():
    scope = measurement_scope(_CHEMBL_FACTS, endpoint="herg")
    assert isinstance(scope, EvidenceScope)
    assert scope.endpoint == "herg"
    assert scope.ligand_id == "CHEMBL1220"
    assert scope.readout == "IC50"
    assert scope.unit == "nM"
    assert scope.comparator == "="
    assert scope.assay_type == "F"


def test_an_unstated_field_stays_unstated_rather_than_being_guessed():
    scope = measurement_scope({"standard_type": "IC50"}, endpoint="herg")
    assert scope.assay_type is None
    assert scope.species is None
    # and that absence does not make it comparable to anything
    assert not scope.comparable_with(EvidenceScope(ligand_id="CHEMBL1220"))


def test_two_records_sharing_no_declared_axis_are_not_comparable():
    """Agreeing on nothing is not agreement -- this is how "both mention
    hERG" turns into a contradiction between unlike things."""
    one = EvidenceScope(assay_type="B")
    other = EvidenceScope(species="human")
    assert one.differences(other) == ()
    assert not one.comparable_with(other)


def test_a_bound_keeps_its_relation_so_it_is_not_read_as_a_measurement():
    bound = measured_value({**_CHEMBL_FACTS, "standard_relation": ">", "standard_value": "30000"})
    assert bound["relation"] == ">"
    assert bound["value"] == pytest.approx(30000.0)
    assert bound["state"] == "measured"


def test_a_record_with_no_value_reports_not_assessed():
    assert measured_value({"standard_type": "IC50"})["state"] == "not_assessed"


def test_the_normalised_record_lists_the_axes_it_actually_declared():
    record = normalised_record(_CHEMBL_FACTS, endpoint="herg")
    assert "assay_type" in record["declared_axes"]
    assert "target_variant" not in record["declared_axes"]
    assert record["measurement"]["unit"] == "nM"


def test_relation_labels_still_offer_insufficient_as_a_real_outcome():
    """"We looked and the record does not settle it" has to stay sayable."""
    assert RelationLabel.INSUFFICIENT.value == "insufficient"
