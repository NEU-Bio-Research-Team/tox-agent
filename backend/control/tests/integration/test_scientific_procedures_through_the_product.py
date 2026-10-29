"""run_scientific_procedure through the registry and the database (#66).

The unit tests prove the four procedures compute the right thing. This proves
the tool is reachable the way a run reaches it: registered only with its flag
on, visible in the right profiles, reading records this session accepted, and
refusing by name when asked for something it does not have.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.application.policy import Actor
from toxagent.domain.errors import InvalidRequest
from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceType
from toxagent.domain.ids import EVIDENCE, new_id
from toxagent.platform.config import PolicySettings
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime.now(timezone.utc)
SESSION = "ses_" + "1" * 32
RUN = "run_" + "6" * 32
ACTOR = Actor(subject_id="user-1")


def _facts(**overrides):
    base = {
        "molecule_chembl_id": "CHEMBL1220",
        "target_chembl_id": "CHEMBL240",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "300",
        "standard_units": "nM",
        "assay_type": "B",
        "document_chembl_id": "DOC1",
    }
    base.update(overrides)
    return base


def _registry(db):
    predictor = StubPredictor().client()
    return build_registry(db, predictor, CreateAnalysis(db, predictor, PolicySettings()),
                          PolicySettings())


async def _seed_records(db, *facts):
    """Accepted evidence in a session, as a search would have left it."""
    from toxagent.domain.session import Session
    from toxagent.application.policy import Actor

    session = Session.create(Actor(subject_id="user-1").subject_id, now=NOW)
    ids = []
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        for index, payload in enumerate(facts):
            record = EvidenceRecord.create(
                session_id=session.id, provider="chembl",
                provider_record_id=f"rec-{index}", source_type=SourceType.DATABASE,
                title=f"Record {index}", retrieved_at=NOW,
                normalized_facts=payload,
            )
            record = replace(record, status=EvidenceStatus.ACCEPTED)
            await uow.evidence.add(record)
            ids.append(record.id)
        await uow.commit()
    return session.id, ids


async def test_the_tool_exists_only_with_its_flag_on(db, monkeypatch):
    monkeypatch.delenv("TOXAGENT_FLAG_SCIENTIFIC_PROCEDURES_V1", raising=False)
    assert "run_scientific_procedure" not in {
        tool.name for tool in _registry(db).visible_for("decision_support")
    }

    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_PROCEDURES_V1", "1")
    visible = {tool.name for tool in _registry(db).visible_for("decision_support")}
    assert "run_scientific_procedure" in visible
    assert "run_scientific_procedure" in {
        tool.name for tool in _registry(db).visible_for("evidence_research")
    }


async def test_it_is_not_offered_where_reading_is_not_allowed(db, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_PROCEDURES_V1", "1")

    for profile in ("analysis", "report_build", "audit_readonly", "report_synthesis"):
        assert "run_scientific_procedure" not in {
            tool.name for tool in _registry(db).visible_for(profile)
        }, profile


async def _run(db, session_id, payload, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_PROCEDURES_V1", "1")
    registry = _registry(db)
    definition = next(t for t in registry.visible_for("decision_support")
                      if t.name == "run_scientific_procedure")
    context = ToolContext(
        session_id=session_id, run_id=RUN, actor=ACTOR, profile="decision_support",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    return await definition.handler(context, definition.input_model(**payload))


async def test_two_rows_of_one_experiment_come_back_as_one_source(db, monkeypatch):
    session_id, ids = await _seed_records(db, _facts(), _facts())

    output = await _run(db, session_id, {
        "procedure": "deduplicate_by_origin", "evidence_ids": ids, "endpoint": "herg",
    }, monkeypatch)

    assert output.model_view["counts"] == {"in": 2, "independent": 1, "duplicates": 1}
    assert "not an independent source" in output.model_view["exclusions"][0]["reason"]
    assert output.provenance["procedure_version"] == "deduplicate-by-origin-v1"
    assert output.provenance["sub_operations"] == 2


async def test_a_micromolar_record_is_normalised_with_its_conversion(db, monkeypatch):
    session_id, ids = await _seed_records(
        db, _facts(standard_value="0.3", standard_units="uM"))

    output = await _run(db, session_id, {
        "procedure": "normalise_measurements", "evidence_ids": ids, "endpoint": "herg",
    }, monkeypatch)

    measurement = output.model_view["facts"]["measurements"][0]
    assert measurement["constraint"]["canonical"] == {
        "low": "300", "high": "300", "low_inclusive": True, "high_inclusive": True,
        "unit": "nM", "dimension": "concentration",
    }
    assert measurement["constraint"]["conversion"]["factor"] == "1000"


async def test_comparing_needs_exactly_two_records(db, monkeypatch):
    session_id, ids = await _seed_records(db, _facts())

    with pytest.raises(InvalidRequest) as refused:
        await _run(db, session_id, {
            "procedure": "compare_measurements", "evidence_ids": ids, "endpoint": "herg",
        }, monkeypatch)

    assert "exactly two" in str(refused.value)


async def test_a_record_this_session_has_not_read_is_refused_by_name(db, monkeypatch):
    session_id, _ids = await _seed_records(db, _facts())
    stranger = new_id(EVIDENCE)

    with pytest.raises(InvalidRequest) as refused:
        await _run(db, session_id, {
            "procedure": "normalise_measurements", "evidence_ids": [stranger],
        }, monkeypatch)

    assert stranger in str(refused.value)


async def test_checking_dependencies_without_a_case_says_so(db, monkeypatch):
    session_id, _ids = await _seed_records(db, _facts())

    with pytest.raises(InvalidRequest) as refused:
        await _run(db, session_id, {"procedure": "check_dependencies"}, monkeypatch)

    assert "no scientific case" in str(refused.value)


async def test_an_unknown_procedure_never_reaches_the_handler(db, monkeypatch):
    """The allowlist is the input model's own enum, so a name outside it is a
    schema refusal before any code runs (§7.2)."""
    session_id, _ids = await _seed_records(db, _facts())

    with pytest.raises(Exception) as refused:
        await _run(db, session_id, {"procedure": "rm -rf", "evidence_ids": []}, monkeypatch)

    assert "procedure" in str(refused.value)
