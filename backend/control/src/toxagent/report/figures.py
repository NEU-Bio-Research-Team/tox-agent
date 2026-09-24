"""Explanation figures: sanitize, hash, store.

Spec sections 5.5 and 11 ("Figure integrity", "Content safety"). Two jobs, both
deterministic:

**Sanitize.** The SVG arrives from the predictor, which is our own service —
but the report's HTML and PDF renderings inline it, and "the source was
trusted" is exactly the reasoning that turns a rendering pipeline into a script
host. An allowlist of elements and attributes is cheap here and impossible to
retrofit once a renderer is shipping. Anything that fetches (``image``,
``foreignObject``, ``use`` with an external href), anything that executes
(``script``, ``on*``) and anything that navigates (``a``) is dropped, not
escaped: an explanation depiction needs none of them.

**Bind the image to what it depicts.** A stored figure carries the endpoint,
the assay and the observation it was drawn from, so the validator can refuse a
figure that has drifted onto the wrong section — eval scenario 12, which is the
failure a reader cannot possibly catch by eye.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final
from xml.etree import ElementTree

from ..domain.attachment import Attachment, RetentionClass
from ..domain.ids import FIGURE, new_id
from ..domain.report import ReportFigure

#: Bumped whenever this module's output for the same input would change. It is
#: recorded on every figure, so a rendering can be traced to the code that drew
#: it (spec section 5.5's ``renderer_version``).
FIGURE_RENDERER_VERSION: Final = "toxagent-figure-sanitizer-v2"

SVG_NS: Final = "http://www.w3.org/2000/svg"

#: Drawing primitives and structure only. Deliberately no ``a``, ``script``,
#: ``image``, ``foreignObject``, ``use``, ``animate*``, ``set`` or ``style``.
_ALLOWED_ELEMENTS: Final[frozenset[str]] = frozenset(
    {
        "svg", "g", "defs", "path", "rect", "circle", "ellipse", "line",
        "polyline", "polygon", "text", "tspan", "title", "desc",
        "linearGradient", "radialGradient", "stop", "clipPath",
    }
)

#: Presentation attributes. No ``on*`` (event handlers), no ``href``/``xlink:href``
#: (fetches and navigations), no ``style`` (which can carry ``url()``).
_ALLOWED_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    {
        "d", "x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "r", "rx", "ry",
        "width", "height", "viewBox", "transform", "points", "fill",
        "fill-opacity", "fill-rule", "stroke", "stroke-width", "stroke-linecap",
        "stroke-linejoin", "stroke-dasharray", "stroke-opacity", "opacity",
        "font-family", "font-size", "font-weight", "font-style", "text-anchor",
        "dominant-baseline", "class", "id", "offset", "stop-color",
        "stop-opacity", "gradientUnits", "clip-path", "xmlns",
        "preserveAspectRatio",
    }
)

#: A ``fill``/``stroke`` may be a colour or a local ``url(#id)`` reference — a
#: remote one is a fetch, which is the thing being prevented.
_REMOTE_URL = re.compile(r"url\(\s*['\"]?\s*(?!#)", re.IGNORECASE)

#: Declarations inside ``style`` that are carried over as presentation
#: attributes. RDKit writes *all* of its paint this way — the white background
#: rect is ``style='fill:#FFFFFF;…'`` and every bond is
#: ``style='stroke:#000000;…'``. v1 dropped ``style`` wholesale, which left a
#: full-canvas ``<rect>`` with no fill: SVG paints that black, over everything
#: (report figures rendered as black boxes, found 2026-09-13). The attribute
#: form is the same paint with none of what made ``style`` dangerous — no
#: ``@import``, no selectors, and each value is checked like any other.
_STYLE_PROPERTIES: Final[frozenset[str]] = frozenset(
    {
        "fill", "fill-opacity", "fill-rule", "stroke", "stroke-width",
        "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "stroke-opacity",
        "opacity", "font-family", "font-size", "font-weight", "font-style",
        "text-anchor", "dominant-baseline",
    }
)

#: A declaration value that could do anything but paint.
_UNSAFE_STYLE_VALUE = re.compile(r"url\(|expression\(|[\\<>@{}]|javascript:", re.IGNORECASE)


def _style_to_attributes(element: ElementTree.Element, style: str) -> None:
    """Move safe ``style`` declarations onto the element as attributes.

    An attribute already present wins: it was set explicitly, and SVG gives a
    ``style`` declaration precedence only because a stylesheet cascade exists,
    which a sanitized figure no longer has.
    """
    for declaration in style.split(";"):
        name, separator, value = declaration.partition(":")
        name, value = name.strip().lower(), value.strip()
        if not separator or name not in _STYLE_PROPERTIES or not value:
            continue
        if _UNSAFE_STYLE_VALUE.search(value):
            continue
        element.attrib.setdefault(name, value)

#: A depiction that outgrows this is not a depiction. Keeps one malformed
#: explanation from putting a megabyte into every rendering of the report.
MAX_FIGURE_BYTES: Final = 512 * 1024


class FigureRejected(ValueError):
    """The bytes could not be made safe to render. Never a partial figure:
    a half-sanitized image would be a picture nobody can account for."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean(element: ElementTree.Element) -> ElementTree.Element | None:
    if _local_name(element.tag) not in _ALLOWED_ELEMENTS:
        return None
    style = element.attrib.pop("style", None)
    if style:
        _style_to_attributes(element, style)
    for name in list(element.attrib):
        local = _local_name(name)
        value = element.attrib[name]
        if local.lower().startswith("on") or local not in _ALLOWED_ATTRIBUTES:
            del element.attrib[name]
            continue
        if _REMOTE_URL.search(value):
            del element.attrib[name]
    kept: list[ElementTree.Element] = []
    for child in list(element):
        element.remove(child)
        cleaned = _clean(child)
        if cleaned is not None:
            kept.append(cleaned)
    for child in kept:
        element.append(child)
    return element


def sanitize_svg(svg: str) -> str:
    """Return an SVG containing only allowlisted elements and attributes.

    Raises ``FigureRejected`` when the input is not parseable SVG at all — a
    figure the server cannot understand is not one it may pass to a renderer.
    """
    if not svg or not svg.strip():
        raise FigureRejected("empty SVG")
    if len(svg.encode("utf-8")) > MAX_FIGURE_BYTES:
        raise FigureRejected(
            f"SVG is larger than the {MAX_FIGURE_BYTES}-byte figure budget"
        )
    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as exc:
        raise FigureRejected(f"not parseable as SVG: {exc}") from exc
    if _local_name(root.tag) != "svg":
        raise FigureRejected(f"root element is {_local_name(root.tag)!r}, not <svg>")
    cleaned = _clean(root)
    if cleaned is None:  # unreachable given the root check, kept for total-ness
        raise FigureRejected("nothing renderable survived sanitization")
    ElementTree.register_namespace("", SVG_NS)
    return ElementTree.tostring(cleaned, encoding="unicode")


def _target_phrase(endpoint: str, task: str | None, predicted_class: str | None) -> str:
    """What the colours are *about*.

    "Positive contribution" is meaningless without naming the thing being
    pushed towards, and a reader who supplies the missing half themselves
    usually supplies "more toxic" (spec XAI-02). Where the predictor named the
    class it explained, the caption says it.
    """
    target = f"{endpoint} / {task}" if task else endpoint
    return f"{target} — {predicted_class}" if predicted_class else target


def caption_for(
    endpoint: str,
    task: str | None,
    *,
    method: str | None,
    predicted_class: str | None = None,
) -> str:
    target = _target_phrase(endpoint, task, predicted_class)
    how = f" ({method})" if method else ""
    toward = predicted_class or "the predicted class"
    return (
        f"Atom-level attribution for {target}{how}. Red marks structure that moved the "
        f"score towards {toward}; green marks structure that moved it away; grey is "
        "near zero. Colour shows what moved the model's score, not a chemical mechanism."
    )


def _named(atoms: list[dict[str, Any]], limit: int = 3) -> str:
    return ", ".join(
        f"{atom.get('symbol', '?')} at index {atom.get('atom_index')}"
        for atom in atoms[:limit]
    )


def alt_text_for(
    endpoint: str,
    task: str | None,
    *,
    top_positive: list[dict[str, Any]],
    top_negative: list[dict[str, Any]] | None = None,
    predicted_class: str | None = None,
) -> str:
    """The picture in words, in both directions.

    Red and green are hard to tell apart for a substantial minority of readers
    and absent entirely for a screen reader, so the alt text carries the same
    two facts the colours do rather than a description of the colours.
    """
    target = _target_phrase(endpoint, task, predicted_class)
    toward = predicted_class or "the predicted class"
    top_negative = top_negative or []
    if not top_positive and not top_negative:
        return (
            f"Structure diagram for the {target} explanation, with no atom carrying a "
            "contribution above the reporting threshold in either direction."
        )
    parts = [f"Structure diagram for the {target} explanation."]
    if top_positive:
        parts.append(f"Atoms pushing most towards {toward}: {_named(top_positive)}.")
    else:
        parts.append(f"No atom pushed towards {toward} above the reporting threshold.")
    if top_negative:
        parts.append(f"Atoms pushing most away from {toward}: {_named(top_negative)}.")
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class StoredFigure:
    figure: ReportFigure
    attachment: Attachment
    object_key: str


def structure_caption_for(canonical_smiles: str, *, preferred_name: str | None = None) -> str:
    named = f"{preferred_name} — " if preferred_name else ""
    return (
        f"{named}2D structure of the analysed compound ({canonical_smiles}). "
        "Identity only: no attribution is shown here."
    )


def structure_alt_text_for(
    canonical_smiles: str, *, preferred_name: str | None = None
) -> str:
    named = preferred_name or "the analysed compound"
    return (
        f"Two-dimensional chemical structure diagram of {named}, drawn from the canonical "
        f"SMILES {canonical_smiles}. No colouring or highlighting."
    )


async def _store_svg(
    *,
    svg: str,
    object_store,
    owner_id: str,
    session_id: str,
    caption: str,
    alt_text: str,
    now: datetime,
    endpoint: str | None,
    task: str | None,
    observation_id: str | None,
    retention_class: RetentionClass,
) -> StoredFigure:
    """Sanitize, hash, write the bytes, and return the metadata row.

    The object key is the content hash, so re-rendering the same figure writes
    the same object rather than accumulating identical blobs, and two figures
    with the same hash are provably the same image.
    """
    cleaned = sanitize_svg(svg)
    data = cleaned.encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    key = f"figures/{digest}.svg"
    await object_store.put(key, data, content_type="image/svg+xml")
    attachment = Attachment.create(
        owner_id=owner_id,
        session_id=session_id,
        media_type="image/svg+xml",
        object_uri=key,
        sha256=digest,
        size_bytes=len(data),
        # Audit: the figure is part of an immutable report and has to outlive
        # the session that produced it.
        retention_class=retention_class,
        now=now,
    )
    figure = ReportFigure(
        figure_id=new_id(FIGURE),
        attachment_id=attachment.id,
        media_type="image/svg+xml",
        caption=caption,
        alt_text=alt_text,
        content_sha256=digest,
        renderer_version=FIGURE_RENDERER_VERSION,
        endpoint=endpoint,
        task=task,
        observation_id=observation_id,
    )
    return StoredFigure(figure=figure, attachment=attachment, object_key=key)


async def store_structure_figure(
    *,
    svg: str,
    object_store,
    owner_id: str,
    session_id: str,
    canonical_smiles: str,
    preferred_name: str | None = None,
    now: datetime,
    retention_class: RetentionClass = RetentionClass.AUDIT,
) -> StoredFigure:
    """A neutral structure figure, bound to no endpoint and no observation.

    Deliberately carries ``endpoint=None`` and ``observation_id=None``: the
    validator refuses a figure whose endpoint disagrees with the section it sits
    in, and an identity picture belongs to the compound rather than to any one
    prediction. Storing it with a borrowed endpoint would make it look like an
    explanation figure to every check that exists to keep those apart (REP-02).
    """
    return await _store_svg(
        svg=svg,
        object_store=object_store,
        owner_id=owner_id,
        session_id=session_id,
        caption=structure_caption_for(canonical_smiles, preferred_name=preferred_name),
        alt_text=structure_alt_text_for(canonical_smiles, preferred_name=preferred_name),
        now=now,
        endpoint=None,
        task=None,
        observation_id=None,
        retention_class=retention_class,
    )


async def store_explanation_figure(
    *,
    svg: str,
    object_store,
    owner_id: str,
    session_id: str,
    endpoint: str,
    task: str | None,
    observation_id: str,
    caption: str,
    alt_text: str,
    now: datetime,
    retention_class: RetentionClass = RetentionClass.AUDIT,
) -> StoredFigure:
    """An explanation figure, bound to the endpoint, assay and observation it
    was drawn from — so the validator can refuse one that has drifted onto the
    wrong section (eval scenario 12)."""
    return await _store_svg(
        svg=svg,
        object_store=object_store,
        owner_id=owner_id,
        session_id=session_id,
        caption=caption,
        alt_text=alt_text,
        now=now,
        endpoint=endpoint,
        task=task,
        observation_id=observation_id,
        retention_class=retention_class,
    )
