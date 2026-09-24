from datetime import datetime, timezone

import pytest

from toxagent.domain.investigation import CaseState, GoalType
from toxagent.domain.session import Session
from toxagent.persistence.investigations import DurableInvestigationRepository

pytestmark = pytest.mark.anyio


async def test_case_reconstructs_without_runtime_transcript(db):
    now = datetime.now(timezone.utc)
    session = Session.create("owner", now=now)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.commit()

    case = CaseState.create(
        session_id=session.id, subject={"canonical_smiles": "CCO"},
        goal=GoalType.EXPLAIN_PREDICTION, questions=("Why this label?",), now=now,
    )
    repository = DurableInvestigationRepository(db)
    await repository.save_case(case, expected_revision=None)

    # A new unit of work stands in for a restarted control process. Nothing is
    # recovered from a runtime session or transcript.
    async with db.unit_of_work() as uow:
        restored = await uow.investigations.get_case(case.id, session_id=session.id)
    assert restored == case
    assert "transcript" not in restored.to_dict()
