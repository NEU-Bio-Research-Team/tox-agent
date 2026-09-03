"""ClinTox provider: availability reporting and wrong-tokenizer protection.

The checkpoint is retained and the provider is written, so the endpoint returns
as soon as a matching tokenizer exists. These tests pin the two behaviours that
matter while it is unavailable: the reason must be specific and actionable, and
a tokenizer from a different training run must never be accepted.
"""
import shutil
from pathlib import Path

import pytest

from toxpred.scientific.artifacts import ArtifactError, ArtifactFile, ArtifactSpec
from toxpred.scientific.providers.clintox_smilesgnn import (
    MODEL_ID,
    TOKENIZER_FILENAME,
    ClinToxSmilesGnnProvider,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO / "models" / "smilesgnn_model"
CONFIG = REPO / "config" / "smilesgnn_config.yaml"
WRONG_TOKENIZER = REPO / "models" / "smilesgnn_multitask_model" / "tokenizer.pkl"


def make_provider(root: Path, config: Path = CONFIG) -> ClinToxSmilesGnnProvider:
    spec = ArtifactSpec(
        model_id=MODEL_ID,
        provider="clintox_smilesgnn",
        capabilities=frozenset({"clintox"}),
        root=root,
        files=(ArtifactFile("best_model.pt", "unused-for-availability"),),
        required=False,
    )
    return ClinToxSmilesGnnProvider(spec, config_path=config)


# --- availability ----------------------------------------------------------

def test_missing_tokenizer_is_reported_specifically():
    available, reason = make_provider(ARTIFACT_DIR).availability()
    assert available is False
    assert TOKENIZER_FILENAME in reason
    assert "69-token" in reason
    assert "train_hybrid.py" in reason


def test_missing_artifact_directory_is_reported(tmp_path):
    available, reason = make_provider(tmp_path / "absent").availability()
    assert available is False
    assert "artifact directory missing" in reason


def test_missing_checkpoint_is_reported(tmp_path):
    (tmp_path / "empty").mkdir()
    available, reason = make_provider(tmp_path / "empty").availability()
    assert available is False
    assert "checkpoint missing" in reason


def test_missing_config_is_reported(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    (root / "best_model.pt").write_bytes(b"x")
    (root / TOKENIZER_FILENAME).write_bytes(b"x")
    available, reason = make_provider(root, config=tmp_path / "nope.yaml").availability()
    assert available is False
    assert "model config missing" in reason


def test_health_carries_the_reason_without_loading():
    health = make_provider(ARTIFACT_DIR).health()
    assert health.model_id == MODEL_ID
    assert health.loaded is False
    assert TOKENIZER_FILENAME in health.detail


def test_load_raises_rather_than_degrading():
    with pytest.raises(ArtifactError, match="tokenizer missing"):
        make_provider(ARTIFACT_DIR).load()


def test_predict_before_load_raises():
    with pytest.raises(ArtifactError, match="before load"):
        make_provider(ARTIFACT_DIR).predict(["CCO"])


# --- the substitution guard ------------------------------------------------

@pytest.mark.skipif(
    not (ARTIFACT_DIR / "best_model.pt").exists() or not WRONG_TOKENIZER.exists(),
    reason="needs the ClinTox checkpoint and a second SMILES tokenizer on disk",
)
def test_tokenizer_from_a_different_run_is_rejected(tmp_path):
    """An 80-token tokenizer against a 69-token checkpoint must fail loudly.

    Silently accepting it would remap every token and produce confident,
    meaningless probabilities.
    """
    root = tmp_path / "smilesgnn_model"
    root.mkdir()
    shutil.copy(ARTIFACT_DIR / "best_model.pt", root / "best_model.pt")
    shutil.copy(WRONG_TOKENIZER, root / TOKENIZER_FILENAME)

    provider = make_provider(root)
    available, _ = provider.availability()
    assert available is True, "the fixture should look loadable before the vocab check"

    with pytest.raises(Exception) as excinfo:
        provider.load()
    message = str(excinfo.value).lower()
    assert any(word in message for word in ("size", "shape", "vocab", "mismatch")), message
