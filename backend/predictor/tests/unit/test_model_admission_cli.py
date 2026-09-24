"""The checkpoint triage CLI is deliberately read-only and safe by default."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "model_admission.py"
spec = importlib.util.spec_from_file_location("model_admission", SCRIPT)
assert spec and spec.loader
model_admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_admission)


def test_scan_marks_unknown_checkpoint_discovered(tmp_path, capsys):
    models = tmp_path / "models"
    checkpoint = models / "candidate" / "best_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a model, only inventory input")
    manifest = tmp_path / "manifest.yaml"
    # A registry must contain at least one declared release; its artifact is
    # deliberately elsewhere so ``candidate`` remains an unadmitted discovery.
    manifest.write_text(
        "schema_version: 1\nmodels_root: .\nmodels:\n"
        "  - model_id: declared\n    provider: fake\n    capabilities: [herg]\n"
        "    artifact_dir: declared\n    required: false\n"
        "    files:\n      - {path: best_model.pt, sha256: " + "0" * 64 + "}\n"
    )

    assert model_admission.scan(models, manifest) == 0
    output = capsys.readouterr().out
    assert "Found 1 checkpoints" in output
    assert "[discovered] candidate/best_model.pt" in output


def test_inspect_hashes_without_unsafe_pickle_load(tmp_path, capsys):
    checkpoint = tmp_path / "untrusted.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")

    assert model_admission.inspect(checkpoint) == 0
    output = capsys.readouterr().out
    assert "sha256:" in output
    # The command reports a failed safe loader; it never retries unsafe pickle.
    assert "safe_load:" in output
