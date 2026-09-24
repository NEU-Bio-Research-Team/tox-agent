"""Sanitized, derived RDKit SVG depictions for numeric XAI v2 artifacts.

**Colour carries the sign (XAI-02).** The first version of this module used a
one-directional purple ramp keyed on ``relative_importance``, which is an
absolute magnitude. Every highlighted atom therefore looked the same kind of
important, and the single most useful thing an attribution knows — whether a
fragment pushed the score *towards* the class being explained or *away* from it
— was not in the picture at all. A reader looking at two equally dark atoms had
no way to tell that one was the reason for the prediction and the other was the
reason it was not higher.

So the ramp is now diverging and keyed on ``signed_contribution``: red towards
the explained class, green away from it, grey for contributions too small to
mean anything. The scale is symmetric about zero — normalised by the largest
absolute contribution in the molecule — so a +0.4 and a −0.4 are equally
saturated, and reversing every sign produces the mirror image rather than a
differently-shaped one.

**Colour is never the only channel.** Red and green are the classic
indistinguishable pair, so the drawing also annotates the strongest
contributors with ``+``/``−`` atom notes and carries an in-image legend naming
what "towards" means. A reader who cannot separate the hues, and a reader using
a screen reader on the alt text, both still get the direction.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from typing import Any

SVG_RENDERER_VERSION = "rdkit-moldraw2d-xai-v2"

#: Bumped with the palette's *meaning*, not merely its hex values. A figure
#: cached under ``purple-sequential-v1`` is an unsigned magnitude picture; read
#: as though it were this palette it would assert directions it never encoded,
#: so the version is part of every figure's provenance.
SVG_PALETTE_VERSION = "signed-red-green-v1"

#: Diverging pair plus a neutral. Chosen for luminance separation as well as
#: hue, so the red and the green stay distinguishable in greyscale and to a
#: reader with a red-green deficiency.
POSITIVE_COLOR = "#D73027"
NEGATIVE_COLOR = "#1A9850"
NEUTRAL_COLOR = "#BDBDBD"

#: Below this fraction of the molecule's strongest contribution, an atom is
#: drawn neutral. Attribution tails are noise, and colouring them makes a
#: picture where everything is slightly meaningful — which reads as everything
#: being slightly implicated.
NEUTRAL_EPSILON = 0.05

#: How many of the strongest contributors per direction get a ``+``/``−`` note.
#: All of them would bury the structure under punctuation.
ANNOTATED_PER_DIRECTION = 3

SVG_NS = "http://www.w3.org/2000/svg"

_LEGEND_HEIGHT = 38
_CANVAS = (560, 360)

_ALLOWED_TAGS = {"svg", "g", "path", "rect", "ellipse", "line", "polygon", "polyline", "text", "tspan"}
_ALLOWED_ATTRS = {"id", "class", "d", "fill", "fill-opacity", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "font-size", "font-family", "font-weight", "x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "rx", "ry", "points", "viewBox", "width", "height", "xmlns", "style"}


def numeric_sha256(atoms: list[dict[str, Any]], bonds: list[dict[str, Any]]) -> str:
    import json
    return hashlib.sha256(json.dumps({"atoms": atoms, "bonds": bonds}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sanitize_svg(svg: str) -> str:
    root = ET.fromstring(svg)
    for element in list(root.iter()):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in _ALLOWED_TAGS:
            parent = next((candidate for candidate in root.iter() if element in list(candidate)), None)
            if parent is not None:
                parent.remove(element)
            continue
        for key, value in list(element.attrib.items()):
            name = key.rsplit("}", 1)[-1]
            if name not in _ALLOWED_ATTRS or "url(" in value.lower() or "javascript:" in value.lower() or "data:" in value.lower():
                del element.attrib[key]
    # The SVG namespace as the *default*, not as an ``ns0:`` prefix.
    # ElementTree invents a prefix unless told otherwise, and while
    # ``<ns0:svg>`` is valid XML, the HTML parser that inlines this figure into
    # a report's self-contained HTML export does not recognise a prefixed
    # element as SVG and renders nothing at all.
    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="unicode")


def _signed(item: dict[str, Any]) -> float:
    """The contribution's signed value, or zero when the provider reported none.

    Zero rather than the magnitude: an unsigned number drawn on a diverging
    scale would be asserted as positive, and inventing a direction is the exact
    error this palette exists to stop.
    """
    value = item.get("signed_contribution")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def signed_color(value: float, scale: float) -> tuple[float, float, float]:
    """Colour for one signed contribution, on a scale symmetric about zero.

    ``scale`` is the largest absolute contribution in the molecule, so the
    strongest atom in either direction is fully saturated and everything else is
    read relative to it. Magnitude becomes saturation by interpolating from the
    neutral grey towards the directional hue, which keeps a weak positive
    visibly *positive* rather than fading it to white where it would be
    indistinguishable from a weak negative.
    """
    neutral = _hex_to_rgb(NEUTRAL_COLOR)
    if scale <= 0:
        return neutral
    fraction = min(1.0, abs(value) / scale)
    if fraction < NEUTRAL_EPSILON:
        return neutral
    target = _hex_to_rgb(POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR)
    return tuple(  # type: ignore[return-value]
        neutral[i] + (target[i] - neutral[i]) * fraction for i in range(3)
    )


def _legend_group(target_class: str | None) -> ET.Element:
    """An in-image key. Inline rather than left to the caption because the SVG
    travels into HTML and PDF exports, and a legend that lives only in the
    surrounding page is a legend the exported figure does not have."""
    toward = target_class or "the predicted class"
    group = ET.Element("g", {"class": "xai-legend"})
    ET.SubElement(group, "line", {
        "x1": "8", "y1": str(_CANVAS[1] + 1), "x2": str(_CANVAS[0] - 8),
        "y2": str(_CANVAS[1] + 1), "stroke": "#CCCCCC", "stroke-width": "1",
    })
    entries = (
        (POSITIVE_COLOR, f"+ increases {toward}"),
        (NEGATIVE_COLOR, f"− decreases {toward}"),
        (NEUTRAL_COLOR, "near zero"),
    )
    x = 10
    for color, label in entries:
        ET.SubElement(group, "rect", {
            "x": str(x), "y": str(_CANVAS[1] + 13), "width": "12", "height": "12",
            "fill": color, "stroke": "#666666", "stroke-width": "0.5",
        })
        text = ET.SubElement(group, "text", {
            "x": str(x + 17), "y": str(_CANVAS[1] + 23),
            "font-family": "sans-serif", "font-size": "11", "fill": "#222222",
        })
        text.text = label
        x += 22 + int(len(label) * 6.1)
    return group


def _with_legend(svg: str, target_class: str | None) -> str:
    """Grow the canvas and append the legend. Returns the input unchanged if the
    SVG is not shaped the way RDKit's is — a figure without a legend is worse
    than one with, but far better than none at all."""
    try:
        root = ET.fromstring(svg)
    except ET.ParseError:
        return svg
    height = _CANVAS[1] + _LEGEND_HEIGHT
    root.set("height", f"{height}px")
    root.set("width", f"{_CANVAS[0]}px")
    root.set("viewBox", f"0 0 {_CANVAS[0]} {height}")
    root.append(_legend_group(target_class))
    return ET.tostring(root, encoding="unicode")


def _annotate_directions(mol, atoms: list[dict[str, Any]], scale: float) -> None:
    """Mark the strongest contributors in each direction with ``+`` or ``−``.

    The redundant channel colour alone cannot provide. Only the strongest few,
    and only those above the neutral threshold: annotating noise would assert a
    direction the numbers do not support.
    """
    ranked = sorted(atoms, key=lambda item: abs(_signed(item)), reverse=True)
    positive = [a for a in ranked if _signed(a) > 0][:ANNOTATED_PER_DIRECTION]
    negative = [a for a in ranked if _signed(a) < 0][:ANNOTATED_PER_DIRECTION]
    # ASCII "-", not U+2212 MINUS SIGN: RDKit's atom-note font has no glyph for
    # the typographic minus and silently draws an empty path, which is worse
    # than no annotation — the redundant channel would look present in the
    # markup and be invisible on the page.
    for group, note in ((positive, "+"), (negative, "-")):
        for item in group:
            if scale <= 0 or abs(_signed(item)) / scale < NEUTRAL_EPSILON:
                continue
            index = item.get("atom_index")
            if index is None or index >= mol.GetNumAtoms():
                continue
            mol.GetAtomWithIdx(int(index)).SetProp("atomNote", note)


def xai_svg(
    canonical_smiles: str,
    atoms: list[dict[str, Any]],
    bonds: list[dict[str, Any]],
    *,
    target_class: str | None = None,
) -> tuple[str, dict[str, str]]:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("cannot depict invalid canonical SMILES")

    atom_values = {int(item["atom_index"]): _signed(item) for item in atoms}
    bond_values = {int(item["bond_index"]): _signed(item) for item in bonds}
    # One scale for atoms and bonds together: highlighting them on separate
    # normalisations would make a bond look as strong as the strongest atom
    # while carrying a fraction of its contribution.
    scale = max((abs(v) for v in (*atom_values.values(), *bond_values.values())), default=0.0)

    _annotate_directions(mol, atoms, scale)
    drawer = rdMolDraw2D.MolDraw2DSVG(*_CANVAS)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_values),
        highlightBonds=list(bond_values),
        highlightAtomColors={i: signed_color(v, scale) for i, v in atom_values.items()},
        highlightBondColors={i: signed_color(v, scale) for i, v in bond_values.items()},
    )
    drawer.FinishDrawing()
    # Legend first, sanitizer last: the legend is ours, but running it through
    # the same allowlist as the provider's bytes keeps one exit path for
    # everything this module emits.
    svg = sanitize_svg(_with_legend(drawer.GetDrawingText(), target_class))
    return svg, {
        "numeric_content_sha256": numeric_sha256(atoms, bonds),
        "renderer_version": SVG_RENDERER_VERSION,
        "palette_version": SVG_PALETTE_VERSION,
        "positive_color": POSITIVE_COLOR,
        "negative_color": NEGATIVE_COLOR,
        "neutral_color": NEUTRAL_COLOR,
        "contribution_scale": f"{scale:.12g}",
        "neutral_epsilon": f"{NEUTRAL_EPSILON:g}",
        "target_class": target_class or "",
    }


#: A plain structure drawing has no attribution in it, so it has its own
#: renderer version: reusing the XAI one would make a neutral picture and a
#: heat-mapped one indistinguishable in a figure's provenance.
STRUCTURE_RENDERER_VERSION = "rdkit-moldraw2d-structure-v1"

_STRUCTURE_CANVAS = (420, 300)


def structure_svg(
    canonical_smiles: str, *, atom_numbering: bool = False
) -> tuple[str, dict[str, str]]:
    """A neutral 2D depiction of the molecule. No highlights, no colour scale.

    The report needs this as a *separate* image from the explanation heat map.
    Substance identity and "which atoms moved the score" are different claims,
    and showing the heat map where the structure belongs invites a reader to
    take the colours as a property of the compound rather than of one model's
    behaviour on one endpoint (REP-02).

    ``atom_numbering`` is off by default: indices are what a contributor table
    refers to, and printing them on the identity figure clutters it for every
    reader who is not cross-referencing one.
    """
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("cannot depict invalid canonical SMILES")
    if atom_numbering:
        for atom in mol.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
    drawer = rdMolDraw2D.MolDraw2DSVG(*_STRUCTURE_CANVAS)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return sanitize_svg(drawer.GetDrawingText()), {
        "renderer_version": STRUCTURE_RENDERER_VERSION,
        "atom_numbering": "true" if atom_numbering else "false",
        "canonical_smiles": canonical_smiles,
    }
