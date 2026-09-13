"""XAI-02: the picture encodes the *direction* of each contribution.

The palette this replaced was a one-directional purple ramp over
``relative_importance``, an absolute magnitude. Two atoms of equal darkness could
be the reason for a prediction and the reason it was not higher, and nothing in
the image separated them. These tests pin the properties that fix has to hold:
sign decides hue, magnitude decides saturation, the scale is symmetric, noise
stays grey, and colour is never the only channel.
"""
from __future__ import annotations

import re

import pytest

from toxpred.application.depiction import (
    NEGATIVE_COLOR,
    NEUTRAL_COLOR,
    NEUTRAL_EPSILON,
    POSITIVE_COLOR,
    SVG_PALETTE_VERSION,
    SVG_RENDERER_VERSION,
    signed_color,
    structure_svg,
    xai_svg,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _fills(svg: str) -> list[tuple[float, float, float]]:
    """Every fill colour in the molecule drawing, legend excluded.

    Only the single strongest contribution reaches the endpoint hex exactly —
    everything else is interpolated towards it — so a test that asserts on
    ``"#D73027" in svg`` would be testing the extreme case and nothing else.
    These assertions classify hues instead.
    """
    drawing = svg.partition('<g class="xai-legend">')[0]
    return [
        _rgb(match) for match in re.findall(r"fill:(#[0-9A-Fa-f]{6})", drawing)
    ]


def _reddish(color: tuple[float, float, float]) -> bool:
    return color[0] > color[1] + 0.08 and color[0] > color[2] + 0.08


def _greenish(color: tuple[float, float, float]) -> bool:
    return color[1] > color[0] + 0.08 and color[1] > color[2] + 0.08


def _greyish(color: tuple[float, float, float]) -> bool:
    return max(color) - min(color) < 0.03


def _drawing(svg: str) -> str:
    """The molecule, without the legend.

    The legend always contains all three swatch colours, so asserting "this
    picture has no red in it" against the whole document would always fail —
    and asserting "it has red" would always pass. Every colour assertion below
    is about the structure.
    """
    head, _, _ = svg.partition('<g class="xai-legend">')
    return head.upper()


def _rgb(hex_value: str) -> tuple[float, float, float]:
    value = hex_value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


# --- the colour function ---------------------------------------------------

def test_the_sign_decides_the_hue():
    assert signed_color(1.0, 1.0) == pytest.approx(_rgb(POSITIVE_COLOR))
    assert signed_color(-1.0, 1.0) == pytest.approx(_rgb(NEGATIVE_COLOR))


def test_the_scale_is_symmetric_about_zero():
    """Equal magnitudes in opposite directions are equally saturated.

    Without this, a reader compares the strength of a positive and a negative
    contribution by comparing two different scales, and the comparison they
    make is not the one the numbers support.
    """
    neutral = _rgb(NEUTRAL_COLOR)
    positive = signed_color(0.4, 1.0)
    negative = signed_color(-0.4, 1.0)
    positive_distance = sum(abs(positive[i] - neutral[i]) for i in range(3))
    negative_distance = sum(abs(negative[i] - neutral[i]) for i in range(3))
    # Different hues, same distance from neutral.
    assert positive != negative
    assert positive_distance == pytest.approx(
        negative_distance, rel=0.35
    ), (positive_distance, negative_distance)


def test_magnitude_becomes_saturation():
    neutral = _rgb(NEUTRAL_COLOR)
    weak = signed_color(0.2, 1.0)
    strong = signed_color(0.9, 1.0)
    assert sum(abs(strong[i] - neutral[i]) for i in range(3)) > sum(
        abs(weak[i] - neutral[i]) for i in range(3)
    )


def test_noise_is_grey_rather_than_faintly_implicated():
    """Below the epsilon there is no direction worth asserting. Colouring the
    tail makes a picture where everything is slightly meaningful, which reads
    as everything being slightly implicated."""
    assert signed_color(NEUTRAL_EPSILON / 2, 1.0) == pytest.approx(_rgb(NEUTRAL_COLOR))
    assert signed_color(-NEUTRAL_EPSILON / 2, 1.0) == pytest.approx(_rgb(NEUTRAL_COLOR))


def test_an_unsigned_contribution_is_not_asserted_as_positive():
    """A provider that reported no sign gets grey, not red. Inventing a
    direction is the exact error the diverging palette exists to stop."""
    svg, _ = xai_svg(
        "CCO",
        [{"atom_index": 0, "symbol": "C", "relative_importance": 0.9}],
        [],
    )
    fills = _fills(svg)
    assert fills, "the atom was not highlighted at all"
    assert all(_greyish(c) for c in fills), fills


def test_a_flat_attribution_has_no_scale_and_stays_neutral():
    assert signed_color(0.0, 0.0) == pytest.approx(_rgb(NEUTRAL_COLOR))


# --- the drawing -----------------------------------------------------------

def _mixed_payload():
    atoms = [
        {"atom_index": 0, "symbol": "C", "signed_contribution": 0.9},
        {"atom_index": 1, "symbol": "C", "signed_contribution": -0.8},
        {"atom_index": 2, "symbol": "O", "signed_contribution": 0.004},
    ]
    bonds = [
        {"bond_index": 0, "signed_contribution": 0.5},
        {"bond_index": 1, "signed_contribution": -0.6},
    ]
    return atoms, bonds


def test_a_mixed_payload_produces_both_a_red_and_a_green_region():
    atoms, bonds = _mixed_payload()
    svg, _ = xai_svg("CCO", atoms, bonds, target_class="hERG blocker")
    fills = _fills(svg)
    assert any(_reddish(c) for c in fills), fills
    assert any(_greenish(c) for c in fills), fills
    # The 0.004 atom is noise on a 0.9 scale and must not be implicated.
    assert any(_greyish(c) for c in fills), fills


def test_flipping_every_sign_swaps_the_colours_and_keeps_the_magnitudes():
    atoms, bonds = _mixed_payload()
    flipped_atoms = [{**a, "signed_contribution": -a["signed_contribution"]} for a in atoms]
    flipped_bonds = [{**b, "signed_contribution": -b["signed_contribution"]} for b in bonds]
    original, meta = xai_svg("CCO", atoms, bonds)
    flipped, flipped_meta = xai_svg("CCO", flipped_atoms, flipped_bonds)
    assert original != flipped
    # Same strongest contribution, so the same normalisation: the picture is a
    # mirror, not a differently-shaped one.
    assert meta["contribution_scale"] == flipped_meta["contribution_scale"]
    assert any(_reddish(c) for c in _fills(flipped))
    assert any(_greenish(c) for c in _fills(flipped))
    # The strongest atom was +0.9 and is now -0.9, so the extreme hue swapped.
    assert POSITIVE_COLOR in _drawing(original)
    assert NEGATIVE_COLOR in _drawing(flipped)


def test_the_legend_names_the_target_class():
    """"Positive" without a label is a sentence the reader finishes themselves,
    and they finish it as "more toxic"."""
    atoms, bonds = _mixed_payload()
    svg, meta = xai_svg("CCO", atoms, bonds, target_class="hERG blocker")
    assert "increases hERG blocker" in svg
    assert "decreases hERG blocker" in svg
    assert meta["target_class"] == "hERG blocker"


def test_the_legend_says_something_true_when_no_class_was_named():
    atoms, bonds = _mixed_payload()
    svg, _ = xai_svg("CCO", atoms, bonds)
    assert "the predicted class" in svg


def test_direction_survives_without_colour():
    """Red and green are the classic indistinguishable pair, so the strongest
    contributors in each direction are annotated with ``+``/``-`` as well.

    Asserted as *drawn* glyphs, not merely as set properties: RDKit's atom-note
    font has no U+2212, and the first version of this used one — emitting a
    ``<path class="note" d="">`` that looked present in the markup and was
    invisible on the page.
    """
    atoms, bonds = _mixed_payload()
    svg, _ = xai_svg("CCO", atoms, bonds, target_class="hERG blocker")
    notes = re.findall(r'<path class="note" d="([^"]*)"', svg)
    assert len(notes) == 2, notes
    assert all(path.strip() for path in notes), notes


def test_only_contributors_above_the_epsilon_are_annotated():
    atoms = [
        {"atom_index": 0, "symbol": "C", "signed_contribution": 1.0},
        {"atom_index": 1, "symbol": "C", "signed_contribution": -0.001},
    ]
    svg, _ = xai_svg("CCO", atoms, [])
    assert len(re.findall(r'<path class="note"', svg)) == 1


def test_the_palette_and_renderer_versions_moved_with_the_meaning():
    """A figure cached under the old purple ramp is an unsigned-magnitude
    picture. Read as though it were this palette it would assert directions it
    never encoded, so both versions are part of a figure's provenance."""
    assert SVG_PALETTE_VERSION == "signed-red-green-v1"
    assert SVG_RENDERER_VERSION == "rdkit-moldraw2d-xai-v2"
    _, meta = xai_svg("CCO", *_mixed_payload())
    assert meta["palette_version"] == SVG_PALETTE_VERSION
    assert meta["renderer_version"] == SVG_RENDERER_VERSION


def test_the_svg_uses_the_default_namespace_so_html_can_inline_it():
    """``<ns0:svg>`` is valid XML and invisible in an HTML document: the HTML
    parser does not recognise a prefixed element as SVG. The report's
    self-contained HTML export inlines these bytes, so this is the difference
    between a figure and a blank space (REP-03)."""
    svg, _ = xai_svg("CCO", *_mixed_payload())
    assert svg.startswith("<svg")
    assert "ns0:" not in svg


def test_the_drawing_carries_no_script_or_remote_fetch():
    svg, _ = xai_svg(ASPIRIN, *_mixed_payload())
    lowered = svg.lower()
    for banned in ("<script", "onload", "javascript:", "data:", "url(", "<image", "xlink"):
        assert banned not in lowered, banned


# --- the neutral structure figure -----------------------------------------

def test_the_structure_figure_is_a_different_image_from_the_heat_map():
    """REP-02: identity and model behaviour are different claims, so they are
    different pictures with different renderer versions."""
    heat, heat_meta = xai_svg(ASPIRIN, *_mixed_payload())
    plain, plain_meta = structure_svg(ASPIRIN)
    assert plain != heat
    assert plain_meta["renderer_version"] != heat_meta["renderer_version"]
    # No legend on a neutral drawing either: there is no scale to explain.
    assert POSITIVE_COLOR not in plain.upper()
    assert NEGATIVE_COLOR not in plain.upper()
    assert "xai-legend" not in plain


def test_an_invalid_smiles_is_refused_rather_than_drawn_empty():
    with pytest.raises(ValueError):
        structure_svg("not-a-molecule")
    with pytest.raises(ValueError):
        xai_svg("not-a-molecule", *_mixed_payload())
