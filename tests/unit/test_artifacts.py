"""Artifact verification: a directory existing is not an artifact."""
import json

import pytest
import yaml

from toxpred.scientific.artifacts import ArtifactError, ArtifactFile, ArtifactSpec, load_manifest, sha256_file


@pytest.fixture
def artifact_dir(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "best_model.pt").write_bytes(b"weights")
    (root / "thresholds.json").write_text(json.dumps({"threshold": 0.4}))
    return root


def spec_for(root, **overrides):
    files = tuple(
        ArtifactFile(f.name, sha256_file(f), f.stat().st_size)
        for f in sorted(root.iterdir())
    )
    kwargs = dict(
        model_id="m1", provider="p1", capabilities=frozenset({"herg"}),
        root=root, files=files,
    )
    kwargs.update(overrides)
    return ArtifactSpec(**kwargs)


def test_verify_passes_for_an_intact_artifact(artifact_dir):
    spec_for(artifact_dir).verify()


def test_missing_file_fails(artifact_dir):
    spec = spec_for(artifact_dir)
    (artifact_dir / "best_model.pt").unlink()
    with pytest.raises(ArtifactError, match="missing file"):
        spec.verify()


def test_corrupted_file_fails_on_checksum(artifact_dir):
    spec = spec_for(artifact_dir)
    (artifact_dir / "best_model.pt").write_bytes(b"weightz")  # same length
    with pytest.raises(ArtifactError, match="checksum mismatch"):
        spec.verify()


def test_truncated_file_fails_on_size(artifact_dir):
    spec = spec_for(artifact_dir)
    (artifact_dir / "best_model.pt").write_bytes(b"w")
    with pytest.raises(ArtifactError, match="size mismatch"):
        spec.verify()


def test_absent_root_fails(tmp_path):
    spec = ArtifactSpec(
        model_id="m1", provider="p1", capabilities=frozenset({"herg"}),
        root=tmp_path / "nope", files=(ArtifactFile("a", "0" * 64),),
    )
    with pytest.raises(ArtifactError, match="not a directory"):
        spec.verify()


def test_empty_directory_is_not_a_valid_artifact(tmp_path):
    """The failure mode behind tox21_ensemble_3_best -> models/dualhead_ensemble3."""
    root = tmp_path / "dualhead_ensemble3"
    root.mkdir()
    (root / "dualhead_metrics.json").write_text("{}")
    spec = ArtifactSpec(
        model_id="ensemble", provider="p1", capabilities=frozenset({"tox21"}),
        root=root, files=(ArtifactFile("best_model.pt", "0" * 64),),
    )
    with pytest.raises(ArtifactError, match="missing file: best_model.pt"):
        spec.verify()


def test_all_problems_are_reported_at_once(artifact_dir):
    spec = spec_for(artifact_dir)
    (artifact_dir / "best_model.pt").unlink()
    (artifact_dir / "thresholds.json").write_text("tampered-but-same-length!!")
    with pytest.raises(ArtifactError) as excinfo:
        spec.verify()
    assert "2 problem(s)" in str(excinfo.value)


# --- manifest parsing ------------------------------------------------------

def write_manifest(tmp_path, payload):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def base_manifest():
    return {
        "schema_version": 1,
        "models_root": ".",
        "models": [{
            "model_id": "m1", "provider": "p1", "capabilities": ["herg"],
            "artifact_dir": "model",
            "files": [{"path": "best_model.pt", "sha256": "0" * 64}],
        }],
    }


def test_manifest_round_trip(tmp_path, artifact_dir):
    path = write_manifest(tmp_path, base_manifest())
    specs = load_manifest(path)
    assert set(specs) == {"m1"}
    assert specs["m1"].capabilities == frozenset({"herg"})


def test_unsupported_schema_version_is_rejected(tmp_path):
    payload = base_manifest() | {"schema_version": 99}
    with pytest.raises(ArtifactError, match="schema_version"):
        load_manifest(write_manifest(tmp_path, payload))


def test_model_with_no_files_is_rejected(tmp_path):
    payload = base_manifest()
    payload["models"][0]["files"] = []
    with pytest.raises(ArtifactError, match="declares no files"):
        load_manifest(write_manifest(tmp_path, payload))


def test_model_with_no_capabilities_is_rejected(tmp_path):
    payload = base_manifest()
    payload["models"][0]["capabilities"] = []
    with pytest.raises(ArtifactError, match="declares no capabilities"):
        load_manifest(write_manifest(tmp_path, payload))


def test_empty_manifest_is_rejected(tmp_path):
    with pytest.raises(ArtifactError, match="declares no models"):
        load_manifest(write_manifest(tmp_path, {"schema_version": 1, "models": []}))


def test_real_manifest_verifies_against_real_artifacts():
    """The shipped manifest must describe the artifacts actually on disk."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    specs = load_manifest(repo / "artifacts" / "predictor-manifest.yaml")
    for spec in specs.values():
        spec.verify()


# --- optional models and declared thresholds -------------------------------

def test_declared_thresholds_are_parsed(tmp_path):
    payload = base_manifest()
    payload["models"][0]["declared_thresholds"] = {"clintox": 0.35}
    payload["models"][0]["required"] = False
    payload["models"][0]["blocked_reason"] = "tokenizer absent"
    specs = load_manifest(write_manifest(tmp_path, payload))
    spec = specs["m1"]
    assert spec.declared_thresholds == {"clintox": 0.35}
    assert spec.required is False
    assert spec.blocked_reason == "tokenizer absent"


def test_real_manifest_declares_clintox_as_optional():
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    specs = load_manifest(repo / "artifacts" / "predictor-manifest.yaml")
    clintox = specs["clintox-smilesgnn-v1"]
    assert clintox.required is False
    assert clintox.declared_thresholds["clintox"] == 0.35
    assert "tokenizer.pkl" in clintox.blocked_reason
    assert specs["herg-tox21-chemberta-v1"].required is True
