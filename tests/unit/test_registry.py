"""Registry behaviour: no fallback, and optional models degrade rather than block."""
import pytest
import yaml

from toxpred.scientific.artifacts import ArtifactError, ArtifactSpec, sha256_file
from toxpred.scientific.registry import ModelHealth, ModelRegistry


class FakeProvider:
    def __init__(self, spec: ArtifactSpec, *, fail: bool = False):
        self.model_id = spec.model_id
        self.capabilities = spec.capabilities
        self._fail = fail
        self._loaded = False

    def load(self):
        if self._fail:
            raise ArtifactError(f"[{self.model_id}] deliberate load failure")
        self._loaded = True

    def health(self):
        return ModelHealth(self.model_id, self._loaded, self.capabilities, "fake")

    def predict(self, canonical_smiles):
        return [{"model_id": self.model_id} for _ in canonical_smiles]


@pytest.fixture
def manifest(tmp_path):
    def build(models):
        for entry in models:
            root = tmp_path / entry["artifact_dir"]
            root.mkdir(exist_ok=True)
            blob = root / "best_model.pt"
            blob.write_bytes(entry["artifact_dir"].encode())
            entry["files"] = [{"path": "best_model.pt", "sha256": sha256_file(blob)}]
        path = tmp_path / "manifest.yaml"
        path.write_text(
            yaml.safe_dump({"schema_version": 1, "models_root": ".", "models": models})
        )
        return path

    return build


def entry(model_id, provider, capabilities, required=True):
    return {
        "model_id": model_id, "provider": provider, "capabilities": capabilities,
        "artifact_dir": model_id, "required": required,
    }


def test_registers_a_valid_model(manifest):
    path = manifest([entry("m1", "p", ["herg"])])
    registry = ModelRegistry.from_manifest(path, {"p": FakeProvider})
    assert registry.describe_capabilities() == {"herg": ["m1"]}
    assert registry.is_ready(["herg"]) == (True, [])


def test_required_model_failing_to_load_raises(manifest):
    path = manifest([entry("m1", "p", ["herg"])])
    with pytest.raises(ArtifactError, match="deliberate load failure"):
        ModelRegistry.from_manifest(
            path, {"p": lambda spec: FakeProvider(spec, fail=True)}
        )


def test_optional_model_failing_does_not_block_the_rest(manifest):
    """The ClinTox case: a declared-but-unavailable model must not fail startup."""
    path = manifest([
        entry("required-model", "p", ["herg"]),
        entry("optional-model", "q", ["clintox"], required=False),
    ])
    registry = ModelRegistry.from_manifest(
        path,
        {"p": FakeProvider, "q": lambda spec: FakeProvider(spec, fail=True)},
    )
    assert registry.is_ready(["herg"]) == (True, [])
    assert registry.describe_capabilities() == {"herg": ["required-model"]}
    unavailable = registry.unavailable()
    assert unavailable["optional-model"]["required"] is False
    assert "deliberate load failure" in unavailable["optional-model"]["reason"]


def test_requesting_an_unavailable_capability_is_not_ready(manifest):
    path = manifest([entry("m1", "p", ["herg"])])
    registry = ModelRegistry.from_manifest(path, {"p": FakeProvider})
    ready, reasons = registry.is_ready(["clintox"])
    assert ready is False
    assert "capability 'clintox' has no provider" in reasons


def test_no_silent_substitution_for_a_missing_capability(manifest):
    """A missing model must raise, never resolve to a different one."""
    path = manifest([entry("m1", "p", ["herg"])])
    registry = ModelRegistry.from_manifest(path, {"p": FakeProvider})
    with pytest.raises(ArtifactError, match="no registered model provides capability"):
        registry.for_capability("clintox")


def test_ambiguous_capability_is_refused(manifest):
    path = manifest([entry("m1", "p", ["tox21"]), entry("m2", "p", ["tox21"])])
    registry = ModelRegistry.from_manifest(path, {"p": FakeProvider})
    with pytest.raises(ArtifactError, match="ambiguous"):
        registry.for_capability("tox21")


def test_capability_mismatch_between_manifest_and_provider_is_refused(manifest):
    path = manifest([entry("m1", "p", ["herg"])])

    def wrong(spec):
        provider = FakeProvider(spec)
        provider.capabilities = frozenset({"tox21"})
        return provider

    with pytest.raises(ArtifactError, match="do not match manifest"):
        ModelRegistry.from_manifest(path, {"p": wrong})


def test_missing_factory_for_a_required_model_raises(manifest):
    path = manifest([entry("m1", "unknown-provider", ["herg"])])
    with pytest.raises(ArtifactError, match="no provider factory"):
        ModelRegistry.from_manifest(path, {"p": FakeProvider})


def test_corrupt_artifact_is_not_registered(manifest, tmp_path):
    path = manifest([entry("m1", "p", ["herg"], required=False)])
    (tmp_path / "m1" / "best_model.pt").write_bytes(b"tampered")
    registry = ModelRegistry.from_manifest(path, {"p": FakeProvider})
    assert registry.describe_capabilities() == {}
    assert "checksum mismatch" in registry.unavailable()["m1"]["reason"]
