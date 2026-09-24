"""The frozen evaluation split must stay frozen and stay clean."""
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "benchmarks" / "manifests" / "eval-split-v1.json"

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="split manifest not built")


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_content_hash_matches(manifest):
    """Detects an edited split — the thing that would silently move the goalposts."""
    recorded = manifest["content_sha256"]
    without = {k: v for k, v in manifest.items() if k != "content_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(without, indent=2, sort_keys=True).encode()
    ).hexdigest()
    assert recomputed == recorded


def test_split_records_how_it_was_produced(manifest):
    for name, dataset in manifest["datasets"].items():
        assert dataset["split_type"] == "scaffold", name
        assert dataset["seed"] == 42, name
        assert "loader" in dataset, name


def test_molecules_are_canonical_and_unique(manifest):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    for name, dataset in manifest["datasets"].items():
        smiles = [r["canonical_smiles"] for r in dataset["test"]]
        assert len(smiles) == len(set(smiles)), f"{name} has duplicate molecules"
        for s in smiles[:200]:
            assert Chem.MolToSmiles(Chem.MolFromSmiles(s)) == s, f"{name}: {s} not canonical"


def test_tox21_missing_labels_are_null_not_zero(manifest):
    """A missing assay must stay missing; coercing it to 0 invents negatives."""
    rows = manifest["datasets"]["tox21"]["test"]
    nulls = sum(1 for r in rows for v in r["labels"].values() if v is None)
    assert nulls > 0, "Tox21 is sparsely labelled; some labels must be null"
    for r in rows:
        for task, v in r["labels"].items():
            assert v is None or v in (0, 1), f"{task}={v!r}"


def test_herg_labels_are_binary(manifest):
    assert {r["label"] for r in manifest["datasets"]["herg"]["test"]} == {0, 1}
