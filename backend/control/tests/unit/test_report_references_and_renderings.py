"""REP-01 and REP-03: a citation a reader can open, in every format.

The failure these pin down was not that citations were unrecorded. ``Claim.
citation_ids`` and ``EvidenceSynthesis.evidence_ids`` were both stored. It was
that nothing turned a stored ``evd_...`` string into a title and a URL: the
artifact carried no resolved snapshot, the renderers printed the raw id, and the
Markdown export linked its figures at a ``figure:`` URI that nothing resolves.
So research that had been performed and validated still reached the reader as an
opaque identifier, and the exports were missing their pictures.
"""
from __future__ import annotations

import zipfile
from datetime import date, datetime, timezone

import pytest

from toxagent.domain.answer import Claim, ClaimKind, Limitation, LimitationCode
from toxagent.domain.evidence import (
    EvidenceRecord,
    EvidenceStatus,
    SourceIdentifier,
    SourceQualityTier,
    SourceType,
)
from toxagent.domain.ids import new_id
from toxagent.domain.report import (
    REQUIRED_SECTION_IDS,
    EvidenceRelation,
    EvidenceSynthesis,
    ReportArtifact,
    ReportFigure,
    ReportReference,
    ReportSection,
    SourceClass,
    SubstanceProfile,
    is_safe_citation_url,
)
from toxagent.report.draft_compiler import citation_order, compile_references
from toxagent.report.renderers import (
    figure_asset_name,
    render_html,
    render_markdown,
    render_markdown_bundle,
)
from toxagent.validation.report.draft_wire import ReportDraftCandidate

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
SESSION = new_id("ses")


# --- fixtures ---------------------------------------------------------------


def _record(
    *,
    title: str,
    url: str | None = "https://europepmc.org/article/MED/12345",
    status: EvidenceStatus = EvidenceStatus.ACCEPTED,
    doi: str | None = "10.1000/example",
) -> EvidenceRecord:
    record = EvidenceRecord.create(
        session_id=SESSION,
        provider="europepmc",
        provider_record_id=title,
        source_type=SourceType.ARTICLE,
        title=title,
        retrieved_at=NOW,
        authors=("Nguyen, T.", "Tran, H."),
        published_at=date(2024, 5, 1),
        canonical_url=url,
        identifier=SourceIdentifier(doi=doi, pmid="12345"),
        source_quality_tier=SourceQualityTier.PRIMARY,
    )
    if status is EvidenceStatus.ACCEPTED:
        return record.to_status(EvidenceStatus.NORMALIZED).to_status(EvidenceStatus.ACCEPTED)
    if status is EvidenceStatus.REJECTED:
        return record.to_status(EvidenceStatus.REJECTED, reason="off topic")
    return record


def _draft(*, first: str, second: str) -> ReportDraftCandidate:
    """A draft citing ``second`` in the executive summary and ``first`` later.

    Deliberately out of "claim order": the number a reader sees has to ascend
    down the page, so the summary's inline citation must become [1] even though
    the other record is attached to an earlier claim in the claims list.
    """
    bodies = {
        "executive_summary": f"Screening summary with an inline citation [@{second}].",
        "external_evidence": "Two records were read; they disagree about the assay.",
    }
    sections = []
    for section_id in REQUIRED_SECTION_IDS:
        source_classes = ["agent_synthesis"]
        if section_id in {"executive_summary", "external_evidence"}:
            source_classes = ["external_evidence"]
        sections.append({
            "section_id": section_id,
            "heading": section_id.replace("_", " ").title(),
            "body_markdown": bodies.get(section_id, "Recorded content."),
            "source_classes": source_classes,
        })
    return ReportDraftCandidate.model_validate({
        "report_build_id": new_id("rpb"),
        "title": "Toxicity Screening Report",
        "sections": sections,
        "claims": [{
            "claim_id": new_id("clm"),
            "kind": "scientific",
            "text": "Published work reports hERG activity for this scaffold.",
            "citation_ids": [first],
        }],
        "evidence_synthesis": [{
            "proposition": "The scaffold is associated with hERG blockade.",
            "relation": "supports",
            "evidence_ids": [first, second],
        }],
    })


def _artifact(
    *,
    references: tuple[ReportReference, ...] = (),
    figures: tuple[ReportFigure, ...] = (),
    body: str = "Recorded content.",
    evidence_synthesis: tuple[EvidenceSynthesis, ...] = (),
    claims: tuple[Claim, ...] = (),
    figure_in_section: str | None = None,
) -> ReportArtifact:
    sections = []
    for section_id in REQUIRED_SECTION_IDS:
        sections.append(ReportSection(
            section_id=section_id,
            heading=section_id.replace("_", " ").title(),
            body_markdown=body if section_id == "executive_summary" else "Recorded content.",
            claim_ids=tuple(c.claim_id for c in claims)
            if section_id == "executive_summary" else (),
            figure_ids=(figure_in_section,)
            if figure_in_section and section_id == "explanation_and_visuals" else (),
            source_classes=(SourceClass.EXTERNAL_EVIDENCE,),
        ))
    return ReportArtifact.create(
        report_build_id=new_id("rpb"),
        session_id=SESSION,
        analysis_id=new_id("ana"),
        title="Toxicity Screening Report",
        subject=SubstanceProfile(canonical_smiles="CCO"),
        sections=tuple(sections),
        figures=figures,
        claims=claims,
        evidence_synthesis=evidence_synthesis,
        limitations=(Limitation(code=LimitationCode.SCREENING_NOT_SAFETY_ASSESSMENT, text="Screening only."),),
        references=references,
        provenance={"run_id": new_id("run")},
        now=NOW,
    )


def _figure() -> ReportFigure:
    return ReportFigure(
        figure_id=new_id("fig"),
        attachment_id=new_id("att"),
        media_type="image/svg+xml",
        caption="Atom-level attribution for herg.",
        alt_text="Structure diagram with the strongest contributors marked.",
        content_sha256="a" * 64,
        renderer_version="toxagent-figure-sanitizer-v1",
    )


def _reference(number: int, *, url: str | None = "https://example.org/a", **kwargs):
    return ReportReference(
        evidence_id=kwargs.pop("evidence_id", new_id("evd")),
        number=number,
        title=kwargs.pop("title", "A published study"),
        provider="europepmc",
        canonical_url=url,
        authors=("Nguyen, T.", "Tran, H."),
        published_at="2024-05-01",
        identifier={"doi": "10.1000/example"},
        source_type="article",
        source_quality_tier="primary",
        retrieved_at=NOW.isoformat(),
        **kwargs,
    )


# --- URL safety -------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "http://example.org/a",
    "https://",
    "",
    None,
    "ftp://example.org/a",
])
def test_an_unsafe_or_unusable_url_never_becomes_a_link(url):
    assert not is_safe_citation_url(url)
    assert _reference(1, url=url).link_url is None


def test_an_https_url_is_linkable():
    assert is_safe_citation_url("https://europepmc.org/article/MED/1")
    assert _reference(1, url="https://europepmc.org/article/MED/1").link_url is not None


def test_a_refused_url_keeps_its_citation_metadata():
    """Refusing the link is not refusing the source. Dropping the citation would
    hide that the claim had a basis at all."""
    reference = _reference(1, url="http://example.org/a")
    assert reference.link_url is None
    assert reference.canonical_url == "http://example.org/a"
    assert reference.title == "A published study"
    assert "Nguyen" in reference.short_form


# --- numbering --------------------------------------------------------------

def test_numbering_follows_the_reader_not_the_claims_list():
    first, second = new_id("evd"), new_id("evd")
    draft = _draft(first=first, second=second)
    # The executive summary's inline token comes first in the document, so it
    # gets [1] even though the other record is the first claim's citation.
    assert citation_order(draft)[:2] == [second, first]


def test_the_same_source_cited_twice_is_numbered_once():
    evidence_id = new_id("evd")
    draft = _draft(first=evidence_id, second=evidence_id)
    assert citation_order(draft) == [evidence_id]


def test_compiled_references_snapshot_the_record_rather_than_its_id():
    first, second = _record(title="First study"), _record(title="Second study")
    draft = _draft(first=first.id, second=second.id)
    references = compile_references(draft, {first.id: first, second.id: second})

    assert [r.number for r in references] == [1, 2]
    by_id = {r.evidence_id: r for r in references}
    assert by_id[first.id].title == "First study"
    assert by_id[first.id].provider == "europepmc"
    assert by_id[first.id].identifier == {"doi": "10.1000/example", "pmid": "12345"}
    assert by_id[first.id].published_at == "2024-05-01"
    assert by_id[first.id].retrieved_at == NOW.isoformat()
    assert all(r.unresolved_reason is None for r in references)


def test_an_unresolvable_citation_is_visible_rather_than_absent():
    missing = new_id("evd")
    resolved = _record(title="Present")
    draft = _draft(first=resolved.id, second=missing)
    references = compile_references(draft, {resolved.id: resolved})

    unresolved = next(r for r in references if r.evidence_id == missing)
    assert unresolved.unresolved_reason
    assert unresolved.number == 1  # still numbered, still findable


def test_a_record_that_stopped_being_citable_says_so():
    rejected = _record(title="Withdrawn", status=EvidenceStatus.REJECTED)
    draft = _draft(first=rejected.id, second=rejected.id)
    reference = compile_references(draft, {rejected.id: rejected})[0]
    assert "rejected" in (reference.unresolved_reason or "")


# --- the artifact's own guarantee -------------------------------------------

def test_an_artifact_cannot_cite_a_source_it_did_not_snapshot():
    """The REP-01 failure, refused by the type. A report holding an evidence id
    with nothing behind it is one whose reader is handed an opaque string."""
    claim = Claim(
        claim_id=new_id("clm"), kind=ClaimKind.SCIENTIFIC,
        text="Published work agrees.", citation_ids=(new_id("evd"),),
    )
    with pytest.raises(ValueError, match="no resolved reference snapshot"):
        _artifact(claims=(claim,))


def test_reference_numbering_must_be_contiguous():
    with pytest.raises(ValueError, match="1..2"):
        _artifact(references=(_reference(1), _reference(3)))


def test_the_reference_snapshot_is_inside_the_content_hash():
    """A citation resolving to a different source is a different report, and a
    hash that could not see the difference would certify it."""
    reference = _reference(1)
    one = _artifact(references=(reference,))
    from dataclasses import replace

    other = _artifact(references=(replace(reference, title="Something else"),))
    assert one.content_sha256 != other.content_sha256


# --- rendering: markdown ----------------------------------------------------

def _cited_artifact():
    reference = _reference(1, title="A published study")
    claim = Claim(
        claim_id=new_id("clm"), kind=ClaimKind.SCIENTIFIC,
        text="Published work agrees.", citation_ids=(reference.evidence_id,),
    )
    return reference, _artifact(
        references=(reference,),
        claims=(claim,),
        body=f"Summary with an inline citation [@{reference.evidence_id}].",
        evidence_synthesis=(
            EvidenceSynthesis(
                synthesis_id=new_id("syn"),
                proposition="The scaffold is associated with hERG blockade.",
                relation=EvidenceRelation.SUPPORTS,
                evidence_ids=(reference.evidence_id,),
            ),
        ),
    )


def test_markdown_turns_a_token_into_the_artifacts_number():
    reference, artifact = _cited_artifact()
    out = render_markdown(artifact)
    assert f"[@{reference.evidence_id}]" not in out
    assert "citation [1]" in out


def test_markdown_lists_the_reference_with_title_authors_and_link():
    reference, artifact = _cited_artifact()
    out = render_markdown(artifact)
    assert "**[1]** A published study" in out
    assert "Nguyen, T." in out
    assert f"<{reference.canonical_url}>" in out
    assert "DOI: 10.1000/example" in out
    assert "retrieved 2026-09-09" in out


def test_markdown_names_each_sections_own_sources():
    _, artifact = _cited_artifact()
    out = render_markdown(artifact)
    assert "Sources for this section: [1] Nguyen, T. et al. (2024)" in out


def test_markdown_figure_links_point_into_the_bundle_not_a_dead_scheme():
    figure = _figure()
    artifact = _artifact(figures=(figure,), figure_in_section=figure.figure_id)
    out = render_markdown(artifact)
    assert "figure:" not in out
    assert f"figures/{figure_asset_name(figure.to_dict())}" in out


def test_the_bundle_contains_the_markdown_and_every_figure_it_links():
    figure = _figure()
    artifact = _artifact(figures=(figure,), figure_in_section=figure.figure_id)
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'
    data = render_markdown_bundle(artifact, figure_svgs={figure.figure_id: svg})

    with zipfile.ZipFile(__import__("io").BytesIO(data)) as archive:
        names = set(archive.namelist())
        assert "report.md" in names
        asset = f"figures/{figure_asset_name(figure.to_dict())}"
        assert asset in names
        # The link in report.md resolves to a file that is actually present.
        assert asset in archive.read("report.md").decode()
        assert archive.read(asset).decode() == svg
        assert "MISSING.txt" not in names


def test_a_bundle_says_which_figures_it_could_not_include():
    figure = _figure()
    artifact = _artifact(figures=(figure,), figure_in_section=figure.figure_id)
    data = render_markdown_bundle(artifact, figure_svgs={})
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as archive:
        assert "MISSING.txt" in archive.namelist()
        assert figure.figure_id in archive.read("MISSING.txt").decode()


def test_the_bundle_is_byte_deterministic():
    """The same artifact must produce the same bytes, or its content hash — and
    therefore any check that two downloads are the same document — is noise."""
    figure = _figure()
    artifact = _artifact(figures=(figure,), figure_in_section=figure.figure_id)
    svg = '<svg xmlns="http://www.w3.org/2000/svg"></svg>'
    assert render_markdown_bundle(artifact, figure_svgs={figure.figure_id: svg}) == (
        render_markdown_bundle(artifact, figure_svgs={figure.figure_id: svg})
    )


# --- rendering: html --------------------------------------------------------

def test_html_citations_are_anchors_into_the_reference_list():
    reference, artifact = _cited_artifact()
    out = render_html(artifact)
    assert f'href="#reference-{reference.evidence_id}"' in out
    assert f'id="reference-{reference.evidence_id}"' in out
    assert ">[1]</a>" in out


def test_html_outbound_links_are_hardened_and_open_in_a_new_tab():
    reference, artifact = _cited_artifact()
    out = render_html(artifact)
    assert (
        f'<a href="{reference.canonical_url}" target="_blank" '
        'rel="noopener noreferrer nofollow">'
    ) in out


def test_html_never_emits_an_unsafe_href_even_when_the_record_had_one():
    reference = _reference(1, url="javascript:alert(1)")
    artifact = _artifact(references=(reference,))
    out = render_html(artifact)
    assert "javascript:" not in out.lower().replace("javascript:alert(1)</code>", "")
    assert 'href="javascript:' not in out
    # The metadata survives; only the anchor is withheld.
    assert "A published study" in out
    assert "not usable as a link" in out


def test_html_escapes_prose_before_inserting_citation_markup():
    """Escape first, then link. The only markup in a paragraph is the anchor
    this renderer built from the artifact's own snapshot."""
    reference = _reference(1)
    artifact = _artifact(
        references=(reference,),
        body=f"<script>alert(1)</script> and a citation [@{reference.evidence_id}].",
    )
    out = render_html(artifact)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert f'href="#reference-{reference.evidence_id}"' in out


def test_html_inlines_a_figure_when_its_bytes_are_supplied():
    figure = _figure()
    artifact = _artifact(figures=(figure,), figure_in_section=figure.figure_id)
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'
    out = render_html(artifact, figure_svgs={figure.figure_id: svg})
    assert svg in out
    assert "[figure unavailable]" not in out
    # Self-contained: nothing is fetched when the file is opened offline. The
    # SVG namespace URI is not a fetch, so the check is on the attributes that
    # would actually load something.
    for fetching in ('src=', '<img', '<link', '@import', 'url(http'):
        assert fetching not in out.lower(), fetching


def test_html_falls_back_to_alt_text_rather_than_a_broken_image():
    figure = _figure()
    artifact = _artifact(figures=(figure,), figure_in_section=figure.figure_id)
    out = render_html(artifact, figure_svgs={})
    assert "[figure unavailable]" in out
    assert figure.alt_text in out


def test_html_carries_print_rules_so_a_figure_is_not_split_across_pages():
    out = render_html(_artifact())
    assert "@media print" in out
    assert "break-inside: avoid" in out


# --- the three formats agree ------------------------------------------------

def test_every_format_uses_the_same_citation_numbering():
    reference, artifact = _cited_artifact()
    markdown = render_markdown(artifact)
    html = render_html(artifact)
    for rendered in (markdown, html):
        assert "[1]" in rendered
        assert reference.title in rendered
        # No format invents its own number, and none leaks the raw id into
        # prose where a reader would have to resolve it themselves.
        assert f"[@{reference.evidence_id}]" not in rendered


def test_a_v1_artifact_still_renders_with_no_reference_snapshot():
    """Older reports were valid when written. Refusing to render them would be a
    worse answer than rendering them exactly as they always were."""
    from dataclasses import replace

    artifact = replace(_artifact(), schema_version="toxagent-report-v1")
    out = render_markdown(artifact)
    assert "Toxicity Screening Report" in out
