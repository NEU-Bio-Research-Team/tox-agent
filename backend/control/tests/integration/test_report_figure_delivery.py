"""REP-02: the figure delivery path, and what it refuses.

Report figures had metadata and no way to fetch them. `ReportBlock` could not
render `figure_ids`, the Markdown export pointed at a `figure:` scheme nothing
resolved, and there was no scoped endpoint at all — so a report's pictures
existed in the object store and nowhere a reader could reach.

Adding the endpoint adds an authorization surface, which is what most of this
file is about. A figure id is not a capability: it has to belong to *this* report
in *this* session, and its bytes have to hash to what the figure claims, because
a report's integrity guarantee covers its pictures and an image is the one part
of a document nobody proofreads.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from toxagent.application.policy import Actor
from toxagent.application.prediction.create_analysis import CreateAnalysis
from toxagent.platform.config import PolicySettings
from toxagent.domain.attachment import Attachment, RetentionClass
from toxagent.domain.ids import new_id
from toxagent.domain.message import Message, Role
from toxagent.domain.report import (
    REQUIRED_SECTION_IDS,
    ReportArtifact,
    ReportFigure,
    ReportSection,
    SubstanceProfile,
)
from toxagent.domain.run import Intent, Lane, Run
from toxagent.domain.session import Session
from toxagent.persistence.object_store import InMemoryObjectStore
from tests.support.api import AUTH, OTHER_AUTH, api_client
from tests.support.predictor import StubPredictor
from tests.support.reports import seed_report_build

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
SVG = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="4"/></svg>'


async def _seed_report(
    db,
    store: InMemoryObjectStore,
    *,
    owner_id: str = "user-1",
    svg: str = SVG,
    corrupt_bytes: bool = False,
) -> tuple[str, str, str]:
    """A session with one report carrying one figure. Returns (session, report,
    figure) ids."""
    session = Session.create(owner_id, now=NOW)
    message = Message.create(session.id, Role.USER, 1, now=NOW)
    run = Run.create(session.id, message.id, Lane.MIXED, Intent.BUILD_REPORT, now=NOW)
    async with db.unit_of_work() as uow:
        await uow.sessions.add(session)
        await uow.messages.add(message)
        await uow.runs.add(run)
        await uow.commit()
    analysis = await CreateAnalysis(db, StubPredictor().client(), PolicySettings()).execute(
        actor=Actor(subject_id=owner_id), session_id=session.id, run_id=run.id,
        smiles="CCO", endpoints=("herg",), owns_run=False,
    )
    build = await seed_report_build(
        db, session_id=session.id, run_id=run.id, analysis_id=analysis.snapshot.id, now=NOW,
    )

    data = svg.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    key = f"figures/{digest}.svg"
    await store.put(key, b"tampered" if corrupt_bytes else data, content_type="image/svg+xml")
    attachment = Attachment.create(
        owner_id=owner_id,
        session_id=session.id,
        media_type="image/svg+xml",
        object_uri=key,
        sha256=digest,
        size_bytes=len(data),
        retention_class=RetentionClass.AUDIT,
        now=NOW,
    )
    figure = ReportFigure(
        figure_id=new_id("fig"),
        attachment_id=attachment.id,
        media_type="image/svg+xml",
        caption="2D structure of the analysed compound (CCO).",
        alt_text="Two-dimensional chemical structure diagram.",
        content_sha256=digest,
        renderer_version="toxagent-figure-sanitizer-v1",
    )
    artifact = ReportArtifact.create(
        report_build_id=build.id,
        session_id=session.id,
        analysis_id=analysis.snapshot.id,
        title="Toxicity Screening Report",
        subject=SubstanceProfile(
            canonical_smiles="CCO", structure_figure_id=figure.figure_id
        ),
        sections=tuple(
            ReportSection(section_id=sid, heading=sid, body_markdown="Recorded content.")
            for sid in REQUIRED_SECTION_IDS
        ),
        figures=(figure,),
        provenance={"run_id": run.id},
        now=NOW,
    )
    async with db.unit_of_work() as uow:
        await uow.attachments.add(attachment)
        await uow.reports.add_artifact(artifact.to_dict(), claim_links=[], evidence_links=[])
        await uow.reports.add_figure(figure, session_id=session.id, created_at=NOW)
        await uow.commit()
    return session.id, artifact.id, figure.figure_id


async def test_a_figure_the_report_carries_is_served(db):
    store = InMemoryObjectStore()
    session_id, report_id, figure_id = await _seed_report(db, store)

    async with api_client(db, StubPredictor(), object_store=store) as client:
        response = await client.get(
            f"/v1/sessions/{session_id}/reports/{report_id}/figures/{figure_id}",
            headers=AUTH,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.text == SVG


async def test_the_response_is_cacheable_and_not_sniffable(db):
    """The artifact is immutable and the object key is the content hash, so this
    URL can only ever return these bytes. `private` because the URL is
    session-scoped and a shared cache would serve it to a second caller."""
    store = InMemoryObjectStore()
    session_id, report_id, figure_id = await _seed_report(db, store)

    async with api_client(db, StubPredictor(), object_store=store) as client:
        response = await client.get(
            f"/v1/sessions/{session_id}/reports/{report_id}/figures/{figure_id}",
            headers=AUTH,
        )

    assert "immutable" in response.headers["cache-control"]
    assert response.headers["cache-control"].startswith("private")
    assert response.headers["etag"].strip('"') == hashlib.sha256(SVG.encode()).hexdigest()
    assert response.headers["x-content-type-options"] == "nosniff"
    # An SVG served as a document could script; served as an image it cannot.
    assert response.headers["content-disposition"].startswith("inline")
    assert "default-src 'none'" in response.headers["content-security-policy"]


async def test_a_figure_from_another_report_is_not_served(db):
    """A figure id is not a capability. Two reports in one session are two
    documents, and one's id must not fetch the other's picture."""
    store = InMemoryObjectStore()
    session_id, report_id, _ = await _seed_report(db, store)
    _, other_report_id, other_figure_id = await _seed_report(db, store)

    async with api_client(db, StubPredictor(), object_store=store) as client:
        response = await client.get(
            f"/v1/sessions/{session_id}/reports/{report_id}/figures/{other_figure_id}",
            headers=AUTH,
        )

    # 404 rather than 403: whether a figure exists elsewhere is not something
    # this endpoint should confirm.
    assert response.status_code == 404
    assert other_report_id != report_id


async def test_another_owners_report_is_not_reachable(db):
    store = InMemoryObjectStore()
    session_id, report_id, figure_id = await _seed_report(db, store)

    async with api_client(db, StubPredictor(), object_store=store) as client:
        response = await client.get(
            f"/v1/sessions/{session_id}/reports/{report_id}/figures/{figure_id}",
            headers=OTHER_AUTH,
        )

    assert response.status_code in (403, 404)


async def test_an_unknown_figure_id_is_a_404_not_a_blank_image(db):
    store = InMemoryObjectStore()
    session_id, report_id, _ = await _seed_report(db, store)

    async with api_client(db, StubPredictor(), object_store=store) as client:
        response = await client.get(
            f"/v1/sessions/{session_id}/reports/{report_id}/figures/{new_id('fig')}",
            headers=AUTH,
        )

    assert response.status_code == 404


async def test_bytes_that_do_not_match_the_recorded_hash_are_refused(db):
    """A report's integrity guarantee covers its pictures. Serving whatever is in
    the store would make the one part of the document nobody proofreads the one
    part nothing checks."""
    store = InMemoryObjectStore()
    session_id, report_id, figure_id = await _seed_report(db, store, corrupt_bytes=True)

    async with api_client(db, StubPredictor(), object_store=store) as client:
        response = await client.get(
            f"/v1/sessions/{session_id}/reports/{report_id}/figures/{figure_id}",
            headers=AUTH,
        )

    assert response.status_code == 404
    assert b"tampered" not in response.content


async def test_the_endpoint_requires_authentication(db):
    store = InMemoryObjectStore()
    session_id, report_id, figure_id = await _seed_report(db, store)

    async with api_client(db, StubPredictor(), object_store=store) as client:
        response = await client.get(
            f"/v1/sessions/{session_id}/reports/{report_id}/figures/{figure_id}"
        )

    assert response.status_code == 401
