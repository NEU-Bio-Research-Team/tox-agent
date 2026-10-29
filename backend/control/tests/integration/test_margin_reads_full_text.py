"""compute_exposure_margin accepts a number written in a record's full text.

Found live on 2026-10-29 (arm E, T5): the IC50 was in Table 1 of the paper
and nowhere in its title, abstract or facts, so the transcription check
refused the margin three times while the model quoted the row it had read.
The captured full text is the source as much as the abstract is -- and a
number in neither is still refused.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.domain.evidence import EvidenceRecord, EvidenceStatus, SourceType
from toxagent.domain.evidence_text import EvidenceFullText, EvidencePassage
from toxagent.domain.session import Session
from toxagent.platform.config import PolicySettings
from toxagent.tools.bootstrap import build_registry
from toxagent.tools.registry import ToolContext
from tests.support.predictor import StubPredictor

pytestmark = pytest.mark.anyio

NOW = datetime.now(timezone.utc)
RUN = "run_" + "7" * 32
ACTOR = Actor(subject_id="user-1")


async def _seed(db) -> tuple[str, str]:
    session = Session.create(ACTOR.subject_id, now=NOW)
    record = replace(EvidenceRecord.create(
        session_id=session.id, provider="europe_pmc", provider_record_id="MED:90011",
        source_type=SourceType.ARTICLE, title="TX-202 hERG IC50 by patch clamp",
        retrieved_at=NOW, abstract_or_excerpt="Free plasma Cmax was 10 nM at the clinical dose.",
    ), status=EvidenceStatus.ACCEPTED)
    full_text = EvidenceFullText.retrieved(
        evidence_id=record.id, session_id=session.id, run_id=RUN,
        passages=(EvidencePassage(
            locator="table:1/row:2", section="Results", kind="table_row",
            text="TX-202 | patch clamp | 0.3 uM", cells=("TX-202", "patch clamp", "0.3 uM"),
        ),),
        section_titles=("Results",), omitted={}, truncated=False, full_text_id=None, now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.evidence.add(record)
        await uow.evidence_text.add_if_absent(full_text)
        await uow.commit()
    return session.id, record.id


async def _margin(db, session_id: str, payload: dict, monkeypatch):
    monkeypatch.setenv("TOXAGENT_FLAG_SCIENTIFIC_PRIMITIVES_V1", "1")
    predictor = StubPredictor().client()
    registry = build_registry(db, predictor, CreateAnalysis(db, predictor, PolicySettings()),
                              PolicySettings())
    definition = next(t for t in registry.visible_for("decision_support")
                      if t.name == "compute_exposure_margin")
    context = ToolContext(
        session_id=session_id, run_id=RUN, actor=ACTOR, profile="decision_support",
        deadline_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    return await definition.handler(context, definition.input_model(**payload))


async def test_an_ic50_written_only_in_a_table_row_is_accepted(db, monkeypatch):
    session_id, evidence_id = await _seed(db)
    result = await _margin(db, session_id, {
        "ic50": {"value": 0.3, "unit": "uM", "source_ref": f"evidence:{evidence_id}"},
        "cmax": {"value": 10, "unit": "nM", "source_ref": f"evidence:{evidence_id}"},
    }, monkeypatch)
    assert result.model_view["margin"] == 30.0


async def test_a_number_in_neither_the_record_nor_its_text_is_still_refused(db, monkeypatch):
    session_id, evidence_id = await _seed(db)
    with pytest.raises(Exception, match="not written in"):
        await _margin(db, session_id, {
            "ic50": {"value": 0.7, "unit": "uM", "source_ref": f"evidence:{evidence_id}"},
            "cmax": {"value": 10, "unit": "nM", "source_ref": f"evidence:{evidence_id}"},
        }, monkeypatch)
