"""What a customer distribution contains, checked rather than remembered (K12).

The handoff plan classifies the tree in a table a person applies. That table
goes stale the first time somebody adds a file, and its two failure modes are
not symmetric: shipping `docs/audit/` discloses internal findings, and
withholding something the setup procedure needs hands the customer a clone
that cannot start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "devops" / "scripts"))

import handoff  # noqa: E402


@pytest.fixture(scope="module")
def classified():
    return handoff.classify(handoff.tracked())


def test_every_tracked_path_is_a_decision(classified):
    """Neither shipping nor withholding is the default. A new file that
    matches no rule fails here, so somebody chooses."""
    _, _, unmatched = classified
    assert unmatched == []


def test_the_setup_procedure_a_customer_follows_is_all_present(classified):
    """The README's seven steps, read back as paths. A distribution missing
    any of these is a clone that cannot start."""
    included = set(classified[0])
    for required in (
        "README.md",
        ".env.example",
        "bin/toxagent",
        "devops/compose/compose.yaml",
        "backend/predictor/registry/models/herg-tox21-chemberta-v1.yaml",
        "backend/control/deploy/Dockerfile",
        "backend/predictor/deploy/Dockerfile",
        "backend/ocr/deploy/Dockerfile",
        "docs/CONFIGURATION.md",
        "docs/OPERATIONS.md",
    ):
        assert required in included, required


def test_internal_material_is_withheld(classified):
    """Audit findings, planning documents and superseded intent."""
    included = set(classified[0])
    withheld = set(classified[1])
    assert not any(path.startswith("docs/audit/") for path in included)
    assert not any(path.startswith("docs/spec/") for path in included)
    assert not any(path.startswith("docs/archive/") for path in included)
    assert not any(path.endswith(".pptx") for path in included)
    assert any(path.startswith("docs/audit/") for path in withheld), (
        "nothing was withheld, so the assertions above pass vacuously"
    )


def test_the_scientific_limitations_travel_with_the_models(classified):
    """A model card that stays behind turns a bounded measurement into an
    unqualified number."""
    included = set(classified[0])
    assert "docs/MODEL_CARD.md" in included
    assert "docs/benchmark-protocol.md" in included
    assert any(path.startswith("docs/artifacts/") for path in included)


def test_no_shipped_document_links_to_a_withheld_one(classified):
    """The quieter failure. Withholding an internal document is right; leaving
    a link to it in a document the customer does receive hands them a dead
    link to something they are not allowed to have."""
    assert handoff.dangling_references(classified[0]) == []


def test_no_credential_shape_ships_unless_it_was_declared(classified):
    assert handoff.secrets_in(classified[0]) == []


def test_every_secret_shape_exemption_still_applies(classified):
    """An exemption for a file that is gone, or is no longer shipped, is an
    exemption nobody notices has stopped meaning anything."""
    included = set(classified[0])
    for path in handoff.secret_shape_exemptions():
        assert (ROOT / path).is_file(), path
        assert path in included, path


def test_every_rule_says_why():
    include, exclude = handoff.rules()
    for rule in include + exclude:
        assert rule.get("why"), rule["glob"]


def test_an_exclusion_wins_over_a_broader_inclusion():
    """`docs/runbooks/**` ships and `docs/audit/**` does not; a rule set where
    the broader one won would make every exclusion depend on no include
    covering it."""
    included, excluded, _ = handoff.classify(
        ["docs/runbooks/x.md", "docs/audit/SYSTEM_ISSUES_VI.md"]
    )
    assert included == ["docs/runbooks/x.md"]
    assert excluded == ["docs/audit/SYSTEM_ISSUES_VI.md"]


def test_the_allowlist_is_json_a_reviewer_can_read():
    document = json.loads((ROOT / "devops" / "handoff_allowlist.json").read_text())
    assert document["schema_version"] == "handoff-allowlist-v1"
    assert document["include"] and document["exclude"]
