"""A sanitized figure keeps its paint and loses only what is unsafe.

Found 2026-09-13: every report figure rendered as a black box. RDKit writes all
of its paint in ``style`` attributes — including the full-canvas white
background ``<rect>`` — and sanitizer v1 dropped ``style`` wholesale. A rect
with no fill is painted black, over the whole drawing.
"""
from __future__ import annotations

from xml.etree import ElementTree

import pytest

from toxagent.report.figures import FIGURE_RENDERER_VERSION, FigureRejected, sanitize_svg

NS = "{http://www.w3.org/2000/svg}"

RDKIT_SHAPED = """<?xml version='1.0' encoding='iso-8859-1'?>
<svg version='1.1' xmlns='http://www.w3.org/2000/svg' xmlns:rdkit='http://www.rdkit.org/xml'
     xml:space='preserve' width='200px' height='150px' viewBox='0 0 200 150'>
<!-- END OF HEADER -->
<rect style='opacity:1.0;fill:#FFFFFF;stroke:none' width='200.0' height='150.0' x='0.0' y='0.0'> </rect>
<path class='bond-0 atom-0 atom-1' d='M 89.5,120.9 L 36.5,120.9' style='fill:none;fill-rule:evenodd;stroke:#000000;stroke-width:2.0px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1' />
<ellipse cx='40' cy='40' rx='10' ry='10' class='atom-0' style='fill:#D73027;fill-rule:evenodd;stroke:#D73027;stroke-width:1.0px' />
</svg>"""


def _elements(svg: str) -> list[ElementTree.Element]:
    return list(ElementTree.fromstring(svg).iter())


def _first(svg: str, tag: str) -> ElementTree.Element:
    return next(e for e in _elements(svg) if e.tag == f"{NS}{tag}")


def test_the_background_rect_stays_white_instead_of_defaulting_to_black():
    rect = _first(sanitize_svg(RDKIT_SHAPED), "rect")
    assert rect.get("fill") == "#FFFFFF"
    assert rect.get("stroke") == "none"
    assert rect.get("style") is None


def test_bonds_and_highlights_keep_their_stroke_and_colour():
    cleaned = sanitize_svg(RDKIT_SHAPED)
    bond = _first(cleaned, "path")
    assert bond.get("fill") == "none"
    assert bond.get("stroke") == "#000000"
    assert bond.get("stroke-width") == "2.0px"
    atom = _first(cleaned, "ellipse")
    assert atom.get("fill") == "#D73027"


@pytest.mark.parametrize(
    "style",
    [
        "fill:url(https://evil.example/x.svg)",
        "fill:url('//evil.example/x')",
        "fill:expression(alert(1))",
        "fill:#fff;behavior:url(x.htc)",
        "fill:red}&lt;/style&gt;&lt;script&gt;",
        "stroke:javascript:alert(1)",
    ],
)
def test_a_declaration_that_could_fetch_or_execute_is_dropped(style):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg'><rect style=\"{style}\" width='1' height='1'/></svg>"
    rect = _first(sanitize_svg(svg), "rect")
    for value in rect.attrib.values():
        assert "url(" not in value.lower()
        assert "expression" not in value.lower()
        assert "javascript" not in value.lower()
    assert rect.get("behavior") is None


def test_an_explicit_attribute_wins_over_a_style_declaration():
    svg = "<svg xmlns='http://www.w3.org/2000/svg'><rect fill='#00FF00' style='fill:#FF0000' width='1' height='1'/></svg>"
    assert _first(sanitize_svg(svg), "rect").get("fill") == "#00FF00"


def test_properties_outside_the_paint_allowlist_do_not_become_attributes():
    svg = "<svg xmlns='http://www.w3.org/2000/svg'><rect style='position:fixed;onload:x;fill:#fff' width='1' height='1'/></svg>"
    rect = _first(sanitize_svg(svg), "rect")
    assert rect.get("position") is None and rect.get("onload") is None
    assert rect.get("fill") == "#fff"


def test_scripts_and_foreign_objects_are_still_removed():
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script>"
        "<foreignObject><div/></foreignObject><rect onclick='x' width='1' height='1'/></svg>"
    )
    cleaned = sanitize_svg(svg)
    tags = {e.tag for e in _elements(cleaned)}
    assert f"{NS}script" not in tags and f"{NS}foreignObject" not in tags
    assert _first(cleaned, "rect").get("onclick") is None


def test_the_version_says_the_output_changed():
    assert FIGURE_RENDERER_VERSION == "toxagent-figure-sanitizer-v2"
    with pytest.raises(FigureRejected):
        sanitize_svg("<html/>")
