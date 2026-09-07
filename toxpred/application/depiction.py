"""Sanitized, derived RDKit SVG depictions for numeric XAI v2 artifacts."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from typing import Any

SVG_RENDERER_VERSION = "rdkit-moldraw2d-xai-v1"
SVG_PALETTE_VERSION = "purple-sequential-v1"
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
    return ET.tostring(root, encoding="unicode")


def xai_svg(canonical_smiles: str, atoms: list[dict[str, Any]], bonds: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("cannot depict invalid canonical SMILES")
    atom_values = {item["atom_index"]: float(item.get("relative_importance", 0.0)) for item in atoms}
    bond_values = {item["bond_index"]: float(item.get("relative_importance", item.get("display_importance", 0.0))) for item in bonds}
    maximum = max([*atom_values.values(), *bond_values.values(), 1e-12])
    color = lambda value: (0.46 + .25 * min(1, value / maximum), 0.26, 0.91)
    drawer = rdMolDraw2D.MolDraw2DSVG(560, 360)
    drawer.DrawMolecule(
        mol, highlightAtoms=list(atom_values), highlightBonds=list(bond_values),
        highlightAtomColors={idx: color(value) for idx, value in atom_values.items()},
        highlightBondColors={idx: color(value) for idx, value in bond_values.items()},
    )
    drawer.FinishDrawing()
    return sanitize_svg(drawer.GetDrawingText()), {
        "numeric_content_sha256": numeric_sha256(atoms, bonds),
        "renderer_version": SVG_RENDERER_VERSION,
        "palette_version": SVG_PALETTE_VERSION,
    }
