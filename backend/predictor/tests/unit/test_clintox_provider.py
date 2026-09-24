"""ClinTox v1 admission: what has to be true, and what is checked.

The checkpoint is retained and the guard is written, so the question this file
pins is not "does it serve" — it does not, and cannot, because the v1 inference
path depended on the retired backend. It is "does the deployment get told the
truth about why".

I07 moved the admission criteria out of prose and into
`ArtifactSpec.tokenizer`. That matters for two reasons these tests hold to:
nothing checked that the checkpoint actually had the 69-token vocabulary the
prose claimed, and because a file named `tokenizer.pkl` proves nothing about
its contents, the only safe implementation was to refuse every tokenizer,
correct ones included.
"""
import shutil
from pathlib import Path

import pytest

from toxpred.scientific.artifacts import (
    ArtifactError, ArtifactFile, ArtifactSpec, TokenizerRequirement, load_manifest, sha256_file,
)
from toxpred.scientific.providers.clintox_smilesgnn import (
    MODEL_ID,
    TOKENIZER_FILENAME,
    ClinToxSmilesGnnProvider,
)

REPO = Path(__file__).resolve().parents[4]
ARTIFACT_DIR = REPO / ".data" / "models" / "smilesgnn_model"
CONFIG = REPO / "backend" / "predictor" / "configs" / "smilesgnn_config.yaml"
WRONG_TOKENIZER = REPO / ".data" / "models" / "smilesgnn_multitask_model" / "tokenizer.pkl"
MANIFEST = REPO / "backend" / "predictor" / "registry" / "models" / "clintox-smilesgnn-v1.yaml"

#: What the shipped manifest declares. Repeated here so a change to it fails a
#: test rather than silently changing what "admitted" means.
REQUIREMENT = TokenizerRequirement(
    relative_path=TOKENIZER_FILENAME,
    vocab_size=69,
    checkpoint_embedding_key="smiles_encoder.token_embedding.weight",
)


def make_provider(
    root: Path, config: Path = CONFIG, tokenizer: TokenizerRequirement | None = REQUIREMENT
) -> ClinToxSmilesGnnProvider:
    spec = ArtifactSpec(
        model_id=MODEL_ID,
        provider="clintox_smilesgnn",
        capabilities=frozenset({"clintox"}),
        root=root,
        files=(ArtifactFile("best_model.pt", "unused-for-availability"),),
        required=False,
        tokenizer=tokenizer,
    )
    return ClinToxSmilesGnnProvider(spec, config_path=config)


# --- the criteria are the manifest's, not the code's -----------------------

def test_the_shipped_manifest_declares_checkable_admission_criteria():
    spec = load_manifest(MANIFEST, models_root=ARTIFACT_DIR.parent)[MODEL_ID]
    assert spec.tokenizer is not None, "the criteria are back in prose"
    assert spec.tokenizer.vocab_size == 69
    assert spec.tokenizer.checkpoint_embedding_key
    # Null on purpose: the training run did not record them, and while they
    # are null nothing can be admitted.
    assert not spec.tokenizer.identity_recorded
    assert spec.blocked_reason


@pytest.mark.needs_artifacts
def test_the_declared_vocabulary_is_checked_against_the_checkpoint():
    """The claim "this is a 69-token model" is read off the weights."""
    assert make_provider(ARTIFACT_DIR).checkpoint_vocab_size() == 69


@pytest.mark.needs_artifacts
def test_a_checkpoint_of_the_wrong_size_is_named_as_the_disagreement(tmp_path):
    """One of the manifest and the checkpoint would be describing a different
    model, and the deployment has to be told which pair disagreed."""
    provider = make_provider(
        ARTIFACT_DIR, tokenizer=TokenizerRequirement(
            relative_path=TOKENIZER_FILENAME, vocab_size=80,
            checkpoint_embedding_key="smiles_encoder.token_embedding.weight",
        ),
    )
    _, unmet = provider.admission_report()
    assert any("69 rows" in reason and "80-token" in reason for reason in unmet), unmet


# --- availability ----------------------------------------------------------

@pytest.mark.needs_artifacts
def test_the_unmet_criteria_are_specific_and_actionable():
    available, reason = make_provider(ARTIFACT_DIR).availability()
    assert available is False
    assert TOKENIZER_FILENAME in reason
    assert "69-token" in reason
    # The criterion that would still block a restored file, stated plainly.
    assert "sha256" in reason
    assert "clintox-smilesgnn-v2" in reason


def test_missing_artifact_directory_is_reported(tmp_path):
    available, reason = make_provider(tmp_path / "absent").availability()
    assert available is False
    assert "artifact directory missing" in reason


def test_missing_checkpoint_is_reported(tmp_path):
    (tmp_path / "empty").mkdir()
    available, reason = make_provider(tmp_path / "empty").availability()
    assert available is False
    assert "checkpoint missing" in reason


def test_missing_config_is_reported(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "best_model.pt").write_bytes(b"x")
    (root / TOKENIZER_FILENAME).write_bytes(b"x")
    available, reason = make_provider(root, config=tmp_path / "nope.yaml").availability()
    assert available is False
    assert "model config missing" in reason


def test_a_spec_with_no_tokenizer_requirement_is_not_admitted(tmp_path):
    """Nothing to check against is not the same as nothing to check."""
    root = tmp_path / "model"
    root.mkdir()
    (root / "best_model.pt").write_bytes(b"x")
    available, reason = make_provider(root, tokenizer=None).availability()
    assert available is False
    assert "declares no tokenizer requirement" in reason


@pytest.mark.needs_artifacts
def test_health_carries_the_reason_without_loading():
    health = make_provider(ARTIFACT_DIR).health()
    assert health.model_id == MODEL_ID
    assert health.loaded is False
    assert TOKENIZER_FILENAME in health.detail


@pytest.mark.needs_artifacts
def test_load_raises_rather_than_degrading():
    with pytest.raises(ArtifactError, match="not admitted"):
        make_provider(ARTIFACT_DIR).load()


def test_predict_before_load_raises():
    with pytest.raises(ArtifactError, match="before load"):
        make_provider(ARTIFACT_DIR).predict(["CCO"])


# --- the substitution guard ------------------------------------------------

@pytest.mark.skipif(
    not (ARTIFACT_DIR / "best_model.pt").exists() or not WRONG_TOKENIZER.exists(),
    reason="needs the ClinTox checkpoint and a second SMILES tokenizer on disk",
)
def test_a_tokenizer_from_a_different_run_fails_the_recorded_identity(tmp_path):
    """The case the hashes exist for.

    An 80-token tokenizer against a 69-token checkpoint would remap every
    token and produce confident, meaningless probabilities. With an identity
    recorded, the substitution is caught by the checksum before anything
    loads — which is what a restored deployment would rely on.
    """
    root = tmp_path / "smilesgnn_model"
    root.mkdir()
    shutil.copy(ARTIFACT_DIR / "best_model.pt", root / "best_model.pt")
    shutil.copy(WRONG_TOKENIZER, root / TOKENIZER_FILENAME)

    provider = make_provider(root, tokenizer=TokenizerRequirement(
        relative_path=TOKENIZER_FILENAME, vocab_size=69,
        checkpoint_embedding_key="smiles_encoder.token_embedding.weight",
        # Stand-in for the hash the training run should have recorded: any
        # value the substituted file does not have exercises the same path.
        sha256="0" * 64,
    ))

    admissible, unmet = provider.admission_report()
    assert admissible is False
    assert any("checksum mismatch" in reason for reason in unmet), unmet
    with pytest.raises(ArtifactError, match="not admitted"):
        provider.load()


@pytest.mark.skipif(
    not (ARTIFACT_DIR / "best_model.pt").exists() or not WRONG_TOKENIZER.exists(),
    reason="needs the ClinTox checkpoint and a second SMILES tokenizer on disk",
)
def test_a_tokenizer_matching_its_recorded_hash_still_does_not_serve(tmp_path):
    """Every criterion met, and v1 still cannot answer a request.

    The inference path depended on the retired backend. Reaching this state
    must produce a refusal that says so, not an endpoint backed by an
    improvised reimplementation wearing v1's name.
    """
    root = tmp_path / "smilesgnn_model"
    root.mkdir()
    shutil.copy(ARTIFACT_DIR / "best_model.pt", root / "best_model.pt")
    shutil.copy(WRONG_TOKENIZER, root / TOKENIZER_FILENAME)

    provider = make_provider(root, tokenizer=TokenizerRequirement(
        relative_path=TOKENIZER_FILENAME, vocab_size=69,
        checkpoint_embedding_key="smiles_encoder.token_embedding.weight",
        sha256=sha256_file(root / TOKENIZER_FILENAME),
    ))

    admissible, unmet = provider.admission_report()
    assert admissible is True, unmet
    with pytest.raises(ArtifactError, match="no inference path"):
        provider.load()
