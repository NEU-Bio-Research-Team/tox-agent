"""Renderings of a report (spec section 10, stage 5).

A rendering is a *view* of the artifact. It adds nothing, drops nothing that
the artifact marked required, and resolves every reference the artifact carries
— which is why the renderers take the compiled artifact and never the draft.

The HTML renderer escapes everything. Report prose has already passed the
content-safety gate, and escaping again here is not redundancy for its own
sake: the validator protects the *artifact*, and this function is what protects
a *reader*, including a reader of an artifact written before a gate existed.
Figures are the one exception, and they are inlined only after passing the same
sanitizer that stored them.

PDF is produced from this module's sanitized HTML through pinned WeasyPrint
66.0. Deployments that omit the ``pdf`` extra fail that one rendering without
losing the canonical artifact or its Markdown/HTML views. It is handed the same
``figure_svgs`` map as HTML: ``render_pdf`` used to call ``render_html(artifact)``
with no figures at all, so every PDF was a document whose every picture said
"[figure unavailable]" (REP-03).

**Citations are the artifact's, in every format.** References come from the
report's own resolved snapshot, numbered once at compile time, so ``[3]`` is the
same source in the in-app view, the Markdown, the HTML and the PDF. A renderer
that numbered independently would produce four documents that cite differently
and agree that they are the same report.
"""
from __future__ import annotations

import html
from typing import Any, Callable, Iterable, Mapping

from ..domain.report import ReportArtifact
from ..validation.citations import CITATION_TOKEN

#: Bumped with REP-01/REP-03: Markdown figure links now resolve into a bundle
#: instead of an unopenable ``figure:`` URI, and every format renders the
#: artifact's numbered reference snapshot.
MARKDOWN_RENDERER_VERSION = "toxagent-markdown-v2"
HTML_RENDERER_VERSION = "toxagent-html-v2"
MARKDOWN_BUNDLE_RENDERER_VERSION = "toxagent-markdown-bundle-v1"
PDF_RENDERER_VERSION = "weasyprint-66.0"

#: Heading text per stable section id. The id is the contract; this is the
#: wording, and changing it changes no data (spec section 5.3).
DEFAULT_HEADINGS: Mapping[str, str] = {
    "executive_summary": "Executive summary",
    "substance_profile": "Substance profile",
    "predictor_results": "Predictor results",
    "explanation_and_visuals": "Explanation and visuals",
    "external_evidence": "External evidence",
    "integrated_interpretation": "Integrated interpretation",
    "conclusions": "Conclusions",
    "recommendations": "Recommendations",
    "limitations": "Limitations",
    "references": "References",
    "provenance_appendix": "Provenance appendix",
}


class RendererUnavailable(RuntimeError):
    """This deployment has no pinned renderer for the requested format."""


def _references(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The artifact's resolved reference snapshot, by evidence id.

    Empty for a ``toxagent-report-v1`` artifact, which carried evidence ids and
    no snapshot. Such a report renders with its citations as bare ids, exactly as
    it always did — the alternative is refusing to render reports that were valid
    when they were written.
    """
    return {r["evidence_id"]: r for r in doc.get("references", ()) or ()}


def _marker(reference: Mapping[str, Any] | None, evidence_id: str) -> str:
    """``[n]`` for a snapshotted source, the bare id for anything else.

    An unresolvable citation is rendered visibly rather than dropped: a sentence
    that silently loses its citation reads as an assertion the report is making
    on its own authority.
    """
    if reference is None:
        return f"[?{evidence_id}]"
    if reference.get("unresolved_reason"):
        return f"[{reference['number']}!]"
    return f"[{reference['number']}]"


def _substitute_markers(body: str, references: Mapping[str, dict[str, Any]]) -> str:
    """Turn ``[@evd_...]`` tokens into the artifact's numbers."""
    return CITATION_TOKEN.sub(
        lambda match: _marker(references.get(match.group(1)), match.group(1)), body or ""
    )


def _section_source_ids(
    doc: Mapping[str, Any], section: Mapping[str, Any], claims: Mapping[str, Any]
) -> list[str]:
    """Which sources this section draws on, in first-appearance order.

    Derived from the section's own prose tokens and its claims' citations rather
    than asserted by the model, so "Sources for this section" cannot list a paper
    the section never cited (REP-01).
    """
    order: list[str] = []

    def add(evidence_id: str) -> None:
        if evidence_id not in order:
            order.append(evidence_id)

    for match in CITATION_TOKEN.finditer(section.get("body_markdown") or ""):
        add(match.group(1))
    for claim_id in section.get("claim_ids", ()):
        claim = claims.get(claim_id)
        if claim:
            for evidence_id in claim.get("citation_ids", ()):
                add(evidence_id)
    if section.get("section_id") == "external_evidence":
        for item in doc.get("evidence_summary", ()):
            for evidence_id in item.get("evidence_ids", ()):
                add(evidence_id)
    return order


def _document(artifact: ReportArtifact | Mapping[str, Any]) -> dict[str, Any]:
    """Both renderers work on the canonical dict, so a report reloaded from the
    database renders identically to one just compiled — the round trip is the
    dict, and rendering from anything else would be a second implementation."""
    return artifact.to_dict() if isinstance(artifact, ReportArtifact) else dict(artifact)


# --- Markdown ---------------------------------------------------------------


def figure_asset_name(figure: Mapping[str, Any]) -> str:
    """The filename a figure takes inside a Markdown bundle.

    Content-hash-suffixed so two figures never collide and the same figure is
    the same file across rebuilds.
    """
    return f"{figure['figure_id']}-{figure['content_sha256'][:12]}.svg"


def render_markdown(
    artifact: ReportArtifact | Mapping[str, Any],
    *,
    figure_svgs: Mapping[str, str] | None = None,
    figure_href: Callable[[Mapping[str, Any]], str] | None = None,
) -> str:
    """``figure_href`` maps a figure to the path this Markdown should link to.

    It defaults to the bundle layout — ``figures/<figure_id>-<hash>.svg`` — which
    is what ``markdown_bundle`` writes. The previous default emitted
    ``figure:<figure_id>``, a scheme nothing resolved, so every downloaded
    Markdown file had a dead image link where a picture belonged (REP-03).
    ``figure_svgs`` is accepted and ignored: Markdown cannot inline SVG, and
    taking the argument keeps one call signature across the three renderers.
    """
    doc = _document(artifact)
    references = _references(doc)
    href = figure_href or (lambda figure: f"figures/{figure_asset_name(figure)}")
    figures = {f["figure_id"]: f for f in doc.get("figures", ())}
    tables = {t["table_id"]: t for t in doc.get("tables", ())}
    gaps = {g["gap_id"]: g for g in doc.get("gaps", ())}
    claims = {c["claim_id"]: c for c in doc.get("claims", ())}

    out: list[str] = [f"# {doc['title']}", ""]
    out += _front_matter(doc)

    for section in doc.get("sections", ()):
        heading = section.get("heading") or DEFAULT_HEADINGS.get(
            section["section_id"], section["section_id"]
        )
        out += [f"## {heading}", ""]
        if section.get("body_markdown", "").strip():
            out += [_substitute_markers(section["body_markdown"].strip(), references), ""]

        for table_id in section.get("table_ids", ()):
            table = tables.get(table_id)
            if table:
                out += _markdown_table(table)

        for figure_id in section.get("figure_ids", ()):
            figure = figures.get(figure_id)
            if figure:
                # A relative path into the bundle. Never a remote URL: a report
                # that fetches on open is a report that reports on its readers.
                out += [
                    f"![{figure['alt_text']}]({href(figure)})",
                    "",
                    f"*{figure['caption']}*",
                    "",
                ]

        for gap_id in section.get("gap_ids", ()):
            gap = gaps.get(gap_id)
            if gap:
                out += [
                    f"> **Gap ({gap['reason']}).** {gap['detail']}",
                    "",
                ]

        out += _section_extras(doc, section["section_id"], claims, references)
        out += _markdown_section_sources(doc, section, claims, references)

    return "\n".join(out).rstrip() + "\n"


def _markdown_section_sources(
    doc: Mapping[str, Any],
    section: Mapping[str, Any],
    claims: Mapping[str, Any],
    references: Mapping[str, dict[str, Any]],
) -> list[str]:
    """The references section lists everything; each other section lists its own,
    so a reader does not have to hold a number in their head while scrolling."""
    if section.get("section_id") == "references":
        return []
    ids = _section_source_ids(doc, section, claims)
    if not ids:
        return []
    parts = []
    for evidence_id in ids:
        reference = references.get(evidence_id)
        marker = _marker(reference, evidence_id)
        parts.append(
            f"{marker} {reference['short_form']}" if reference else marker
        )
    return [f"*Sources for this section: {'; '.join(parts)}*", ""]


def _front_matter(doc: Mapping[str, Any]) -> list[str]:
    status = doc.get("status", "")
    lines = [
        f"*Report `{doc['report_id']}` · analysis `{doc['analysis_id']}` · "
        f"status **{status}** · content hash `{doc['content_sha256'][:12]}`*",
        "",
    ]
    if status == "completed_with_gaps":
        # Said before anything else, because a partial report read as a
        # complete one is the single most consequential misreading available.
        lines += [
            "> **This report is incomplete.** Some requested content could not be "
            "produced; each affected section states what is missing and why.",
            "",
        ]
    return lines


def _section_extras(
    doc: Mapping[str, Any],
    section_id: str,
    claims: Mapping[str, Any],
    references: Mapping[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Structured content the artifact holds outside section bodies.

    Conclusions, recommendations, limitations and references are typed lists,
    not prose, precisely so that a renderer cannot drop a limitation or lose a
    recommendation's basis; they are laid out here from those lists.
    """
    out: list[str] = []
    if section_id == "conclusions":
        for item in doc.get("conclusions", ()):
            scope = (
                "Integrated screening interpretation"
                if item.get("is_integrated")
                else f"{item.get('endpoint')}" + (f" / {item['task']}" if item.get("task") else "")
            )
            out += [f"**{scope}.** {item['text']}", ""]
            out += [f"  - Basis: {_claim_refs(item.get('basis_claim_ids', ()), claims)}", ""]
    elif section_id == "recommendations":
        for item in doc.get("recommendations", ()):
            out += [
                f"**{item['priority'].title()} · {item['action_category']}.** {item['text']}",
                "",
                f"  - Rationale: {item['rationale']}",
            ]
            if item.get("conditions"):
                out += [f"  - Conditions: {item['conditions']}"]
            out += [f"  - Basis: {_claim_refs(item.get('basis_claim_ids', ()), claims)}", ""]
    elif section_id == "limitations":
        for item in doc.get("limitations", ()):
            out += [f"- **{item['code']}** — {item['text']}"]
        out += [""]
    elif section_id == "external_evidence":
        refs = references or {}
        for item in doc.get("evidence_summary", ()):
            context = ", ".join(
                f"{label}: {item[key]}"
                for key, label in (
                    ("endpoint", "endpoint"), ("assay", "assay"),
                    ("organism", "organism"), ("dose_context", "dose"),
                )
                if item.get(key)
            )
            # Numbers, not raw ids: an ``evd_`` string tells a reader nothing
            # and cannot be looked up in the references list.
            markers = " ".join(
                _marker(refs.get(e), e) for e in item.get("evidence_ids", ())
            )
            out += [
                f"- **{item['relation']}** — {item['proposition']}"
                + (f" ({context})" if context else "")
                + (f" {markers}" if markers else "")
            ]
        out += [""]
    elif section_id == "references":
        out += _markdown_references(references or {})
    elif section_id == "provenance_appendix":
        provenance = doc.get("provenance", {})
        for key in sorted(provenance):
            out += [f"- **{key}**: `{provenance[key]}`"]
        out += [""]
    return out


def _markdown_references(references: Mapping[str, dict[str, Any]]) -> list[str]:
    """The numbered list, in the artifact's own numbering."""
    if not references:
        return []
    out: list[str] = []
    for reference in sorted(references.values(), key=lambda r: r["number"]):
        authors = ", ".join(reference.get("authors") or ()) or "—"
        year = (reference.get("published_at") or "")[:4]
        bits = [f"**[{reference['number']}]** {reference['title']}", authors]
        if year:
            bits.append(year)
        bits.append(reference.get("provider") or "unknown provider")
        identifier = ", ".join(
            f"{key.upper()}: {value}"
            for key, value in (reference.get("identifier") or {}).items()
        )
        if identifier:
            bits.append(identifier)
        link = reference.get("link_url")
        if link:
            bits.append(f"<{link}>")
        elif reference.get("canonical_url"):
            # The URL is recorded but not linkable. Saying so beats a link a
            # reader clicks and a renderer refused to make safe.
            bits.append(f"URL not an HTTPS link: `{reference['canonical_url']}`")
        if reference.get("retrieved_at"):
            bits.append(f"retrieved {reference['retrieved_at'][:10]}")
        line = " · ".join(bits)
        if reference.get("unresolved_reason"):
            line += f" — **unresolved:** {reference['unresolved_reason']}"
        out.append(f"- {line}")
    return out + [""]


def _claim_refs(claim_ids: Iterable[str], claims: Mapping[str, Any]) -> str:
    parts = []
    for claim_id in claim_ids:
        claim = claims.get(claim_id)
        parts.append(f"`{claim_id}`" if claim is None else f"`{claim_id}` ({claim['kind']})")
    return ", ".join(parts) if parts else "*none recorded*"


def _markdown_table(table: Mapping[str, Any]) -> list[str]:
    out = [f"**{table['title']}**", ""]
    columns = table["columns"]
    out.append("| " + " | ".join(columns) + " |")
    out.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in table.get("rows", ()):
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    out.append("")
    return out


# --- HTML -------------------------------------------------------------------


_STYLE = """
:root { color-scheme: light dark; }
body { font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 46rem; margin: 2rem auto; padding: 0 1rem; }
h1, h2 { line-height: 1.25; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid currentColor; padding: .4rem .6rem; text-align: left; }
figure { margin: 1.5rem 0; }
figcaption { font-size: .9em; opacity: .8; }
blockquote.gap { border-left: 4px solid currentColor; margin: 1rem 0;
                 padding: .5rem 1rem; opacity: .9; }
.meta { font-size: .9em; opacity: .75; }
.limitation code { font-weight: 600; }
.sources { font-size: .85em; opacity: .8; }
ol.references { padding-left: 1.4rem; }
ol.references li { margin: .5rem 0; }
.unresolved { font-weight: 600; }
a.citation { text-decoration: none; font-weight: 600; }
@media print {
  a.citation::after { content: ""; }
  figure, table, tr, blockquote { break-inside: avoid; page-break-inside: avoid; }
  h2 { break-after: avoid; page-break-after: avoid; }
  figcaption, caption { break-before: avoid; page-break-before: avoid; }
}
"""


def render_html(
    artifact: ReportArtifact | Mapping[str, Any],
    *,
    figure_svgs: Mapping[str, str] | None = None,
) -> str:
    """``figure_svgs`` maps figure_id -> already-sanitized SVG source.

    A figure with no entry renders as its alt text rather than as a broken
    image: the report still says what the picture would have said, which is the
    part a reader actually needs.
    """
    doc = _document(artifact)
    figure_svgs = figure_svgs or {}
    figures = {f["figure_id"]: f for f in doc.get("figures", ())}
    tables = {t["table_id"]: t for t in doc.get("tables", ())}
    gaps = {g["gap_id"]: g for g in doc.get("gaps", ())}
    claims = {c["claim_id"]: c for c in doc.get("claims", ())}
    references = _references(doc)

    e = html.escape
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="' + e(doc.get("report_language", "en")) + '">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{e(doc['title'])}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{e(doc['title'])}</h1>",
        f'<p class="meta">Report <code>{e(doc["report_id"])}</code> · analysis '
        f'<code>{e(doc["analysis_id"])}</code> · status <strong>{e(doc["status"])}</strong> · '
        f'content hash <code>{e(doc["content_sha256"][:12])}</code></p>',
    ]
    if doc.get("status") == "completed_with_gaps":
        parts.append(
            '<blockquote class="gap"><strong>This report is incomplete.</strong> Some '
            "requested content could not be produced; each affected section states what is "
            "missing and why.</blockquote>"
        )

    for section in doc.get("sections", ()):
        heading = section.get("heading") or DEFAULT_HEADINGS.get(
            section["section_id"], section["section_id"]
        )
        parts.append(f'<section id="{e(section["section_id"])}">')
        parts.append(f"<h2>{e(heading)}</h2>")
        body = section.get("body_markdown", "").strip()
        if body:
            # Paragraph-split and escaped, *then* citation markers are inserted
            # as anchors. That order matters: escaping first means the prose can
            # never contribute markup, and the only HTML in a paragraph is the
            # link this renderer built from the artifact's own reference
            # snapshot — never a URL the model wrote.
            for paragraph in body.split("\n\n"):
                parts.append(f"<p>{_html_with_citations(paragraph.strip(), references)}</p>")

        for table_id in section.get("table_ids", ()):
            table = tables.get(table_id)
            if table:
                parts.append(_html_table(table))

        for figure_id in section.get("figure_ids", ()):
            figure = figures.get(figure_id)
            if not figure:
                continue
            svg = figure_svgs.get(figure_id)
            body_html = (
                svg if svg
                else f'<p class="meta">[figure unavailable] {e(figure["alt_text"])}</p>'
            )
            parts.append(
                "<figure>"
                f'<div role="img" aria-label="{e(figure["alt_text"])}">{body_html}</div>'
                f"<figcaption>{e(figure['caption'])}</figcaption>"
                "</figure>"
            )

        for gap_id in section.get("gap_ids", ()):
            gap = gaps.get(gap_id)
            if gap:
                parts.append(
                    f'<blockquote class="gap"><strong>Gap ({e(gap["reason"])}).</strong> '
                    f"{e(gap['detail'])}</blockquote>"
                )

        parts += _html_section_extras(doc, section["section_id"], references)
        parts += _html_section_sources(doc, section, claims, references)
        parts.append("</section>")

    parts += ["</body>", "</html>"]
    return "\n".join(parts)


def _html_with_citations(text: str, references: Mapping[str, dict[str, Any]]) -> str:
    """Escaped prose whose ``[@evd_...]`` tokens have become in-page anchors.

    ``rel="noopener noreferrer nofollow"`` and ``target="_blank"`` are on the
    reference-list link rather than here: this anchor jumps within the document,
    so a reader lands on the full citation and chooses whether to leave.
    """
    escaped = html.escape(text)

    def link(match) -> str:
        evidence_id = match.group(1)
        reference = references.get(evidence_id)
        marker = html.escape(_marker(reference, evidence_id))
        if reference is None:
            return f'<span class="unresolved">{marker}</span>'
        return (
            f'<a class="citation" href="#reference-{html.escape(evidence_id)}">{marker}</a>'
        )

    return CITATION_TOKEN.sub(link, escaped)


def _html_section_sources(
    doc: Mapping[str, Any],
    section: Mapping[str, Any],
    claims: Mapping[str, Any],
    references: Mapping[str, dict[str, Any]],
) -> list[str]:
    if section.get("section_id") == "references":
        return []
    ids = _section_source_ids(doc, section, claims)
    if not ids:
        return []
    e = html.escape
    items = []
    for evidence_id in ids:
        reference = references.get(evidence_id)
        marker = e(_marker(reference, evidence_id))
        if reference is None:
            items.append(f'<span class="unresolved">{marker}</span>')
        else:
            items.append(
                f'<a class="citation" href="#reference-{e(evidence_id)}">{marker}</a> '
                f"{e(reference['short_form'])}"
            )
    return [f'<p class="sources">Sources for this section: {"; ".join(items)}</p>']


def _html_references(references: Mapping[str, dict[str, Any]]) -> list[str]:
    """The numbered list, with the only outbound links in the document.

    A link is produced only for an HTTPS ``link_url`` the artifact already
    vetted; ``javascript:`` and ``data:`` never reach here because
    ``ReportReference.link_url`` refuses them, and a refused URL still shows its
    metadata so a reader can find the source by hand (REP-01).
    """
    if not references:
        return []
    e = html.escape
    out = ['<ol class="references">']
    for reference in sorted(references.values(), key=lambda r: r["number"]):
        anchor = e(reference["evidence_id"])
        bits = [f"<strong>{e(reference['title'])}</strong>"]
        authors = ", ".join(reference.get("authors") or ())
        if authors:
            bits.append(e(authors))
        year = (reference.get("published_at") or "")[:4]
        if year:
            bits.append(e(year))
        bits.append(e(reference.get("provider") or "unknown provider"))
        identifier = ", ".join(
            f"{key.upper()}: {value}"
            for key, value in (reference.get("identifier") or {}).items()
        )
        if identifier:
            bits.append(e(identifier))
        link = reference.get("link_url")
        if link:
            bits.append(
                f'<a href="{e(link)}" target="_blank" rel="noopener noreferrer nofollow">'
                f"{e(link)}</a>"
            )
        elif reference.get("canonical_url"):
            bits.append(
                "URL not usable as a link: <code>"
                + e(reference["canonical_url"])
                + "</code>"
            )
        if reference.get("retrieved_at"):
            bits.append(f"retrieved {e(reference['retrieved_at'][:10])}")
        line = " · ".join(bits)
        if reference.get("unresolved_reason"):
            line += (
                f' <span class="unresolved">unresolved: '
                f"{e(reference['unresolved_reason'])}</span>"
            )
        out.append(f'<li id="reference-{anchor}" value="{reference["number"]}">{line}</li>')
    out.append("</ol>")
    return out


def _html_section_extras(
    doc: Mapping[str, Any],
    section_id: str,
    references: Mapping[str, dict[str, Any]] | None = None,
) -> list[str]:
    e = html.escape
    out: list[str] = []
    if section_id == "conclusions":
        for item in doc.get("conclusions", ()):
            scope = (
                "Integrated screening interpretation"
                if item.get("is_integrated")
                else str(item.get("endpoint") or "")
                + (f" / {item['task']}" if item.get("task") else "")
            )
            out.append(
                f"<p><strong>{e(scope)}.</strong> {e(item['text'])}<br>"
                f'<span class="meta">Basis: '
                f"{e(', '.join(item.get('basis_claim_ids', ())) or 'none recorded')}</span></p>"
            )
    elif section_id == "recommendations":
        for item in doc.get("recommendations", ()):
            out.append(
                f"<p><strong>{e(item['priority'].title())} · "
                f"{e(item['action_category'])}.</strong> {e(item['text'])}<br>"
                f'<span class="meta">Rationale: {e(item["rationale"])}'
                + (f" · Conditions: {e(item['conditions'])}" if item.get("conditions") else "")
                + f" · Basis: {e(', '.join(item.get('basis_claim_ids', ())))}</span></p>"
            )
    elif section_id == "limitations":
        out.append("<ul>")
        for item in doc.get("limitations", ()):
            out.append(
                f'<li class="limitation"><code>{e(item["code"])}</code> — {e(item["text"])}</li>'
            )
        out.append("</ul>")
    elif section_id == "external_evidence":
        snapshot = references or {}
        out.append("<ul>")
        for item in doc.get("evidence_summary", ()):
            markers = " ".join(
                (
                    f'<a class="citation" href="#reference-{e(evidence_id)}">'
                    f"{e(_marker(snapshot.get(evidence_id), evidence_id))}</a>"
                    if evidence_id in snapshot
                    else f'<span class="unresolved">'
                    f"{e(_marker(None, evidence_id))}</span>"
                )
                for evidence_id in item.get("evidence_ids", ())
            )
            out.append(
                f"<li><strong>{e(item['relation'])}</strong> — {e(item['proposition'])}"
                + (f" {markers}" if markers else "")
                + "</li>"
            )
        out.append("</ul>")
    elif section_id == "references":
        out += _html_references(references or {})
    elif section_id == "provenance_appendix":
        provenance = doc.get("provenance", {})
        out.append("<ul>")
        for key in sorted(provenance):
            out.append(f"<li><strong>{e(key)}</strong>: <code>{e(str(provenance[key]))}</code></li>")
        out.append("</ul>")
    return out


def _html_table(table: Mapping[str, Any]) -> str:
    e = html.escape
    head = "".join(f"<th>{e(str(c))}</th>" for c in table["columns"])
    body = "".join(
        "<tr>" + "".join(f"<td>{e(str(cell))}</td>" for cell in row) + "</tr>"
        for row in table.get("rows", ())
    )
    return (
        f"<table><caption>{e(table['title'])}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def render_pdf(
    artifact: ReportArtifact | Mapping[str, Any],
    *,
    figure_svgs: Mapping[str, str] | None = None,
    **_: Any,
) -> bytes:
    """PDF from this module's own sanitized HTML — including its figures.

    ``figure_svgs`` used to be dropped on the floor here: this function called
    ``render_html(artifact)`` positionally and every PDF came out with
    "[figure unavailable]" wherever a picture belonged (REP-03).
    """
    try:
        import weasyprint
    except ImportError as exc:
        raise RendererUnavailable(
            "PDF rendering requires the pinned toxagent-control[pdf] dependency"
        ) from exc
    version = getattr(weasyprint, "__version__", "")
    if version != "66.0":
        raise RendererUnavailable(
            f"PDF renderer version {version!r} is not the pinned 66.0"
        )
    return weasyprint.HTML(
        string=render_html(artifact, figure_svgs=figure_svgs)
    ).write_pdf()


# --- Markdown bundle --------------------------------------------------------


def render_markdown_bundle(
    artifact: ReportArtifact | Mapping[str, Any],
    *,
    figure_svgs: Mapping[str, str] | None = None,
) -> bytes:
    """``report.md`` plus a ``figures/`` directory, as a zip.

    The plain Markdown rendering links its images at ``figures/<name>.svg``. On
    its own that is a relative path into a directory the reader does not have, so
    the bundle is what makes the Markdown actually openable — the alternative
    considered was a figure URL requiring a session token, which is not a
    document somebody can keep (REP-03).

    A figure whose bytes could not be resolved is listed in ``MISSING.txt``
    rather than omitted, so a bundle never quietly has fewer pictures than the
    report it came from.
    """
    import io
    import zipfile

    doc = _document(artifact)
    figure_svgs = figure_svgs or {}
    buffer = io.BytesIO()
    # Deterministic: fixed timestamps and sorted members, so the same artifact
    # produces byte-identical bytes and therefore the same content hash. A
    # rendering whose hash changed on every rebuild would make the export
    # manifest useless for checking that two downloads are the same document.
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        def write(name: str, data: bytes) -> None:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)

        write("report.md", render_markdown(artifact).encode("utf-8"))
        missing: list[str] = []
        for figure in doc.get("figures", ()) or ():
            svg = figure_svgs.get(figure["figure_id"])
            if svg is None:
                missing.append(
                    f"{figure['figure_id']}: bytes unavailable — {figure['alt_text']}"
                )
                continue
            write(f"figures/{figure_asset_name(figure)}", svg.encode("utf-8"))
        if missing:
            write(
                "MISSING.txt",
                (
                    "These figures are referenced by report.md and their bytes could not "
                    "be resolved at export time:\n\n" + "\n".join(missing) + "\n"
                ).encode("utf-8"),
            )
    return buffer.getvalue()
