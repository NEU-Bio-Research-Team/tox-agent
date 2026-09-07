"""Golden regression: the ported provider must reproduce the pre-refactor baseline.

Requires the real checkpoint plus torch/transformers, so it is skipped when the
artifact or the runtime is unavailable. The baseline was captured by
`benchmarks/capture_baseline.py` at commit e6882b2, before any code moved.

Tolerance: 1e-6 on the same CPU container, per the plan's Phase 3 gate.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "benchmarks" / "golden" / "baseline_predictions.json"
MANIFEST = REPO / "artifacts" / "predictor-manifest.yaml"
TOLERANCE = 1e-6

pytestmark = pytest.mark.golden


def _requirements_met() -> tuple[bool, str]:
    if not GOLDEN.exists():
        return False, "baseline not captured — run benchmarks/capture_baseline.py"
    if not MANIFEST.exists():
        return False, "artifact manifest missing"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        return False, f"runtime unavailable: {exc}"
    from toxpred.scientific.artifacts import ArtifactError, load_manifest

    try:
        for spec in load_manifest(MANIFEST).values():
            spec.verify()
    except ArtifactError as exc:
        return False, f"artifact unavailable: {exc}"
    return True, ""


_OK, _REASON = _requirements_met()
pytestmark = [pytest.mark.golden, pytest.mark.skipif(not _OK, reason=_REASON)]


@pytest.fixture(scope="module")
def predictions():
    from toxpred.scientific.providers.herg_tox21_chemberta import factory
    from toxpred.scientific.registry import ModelRegistry

    registry = ModelRegistry.from_manifest(MANIFEST, {"herg_tox21_chemberta": factory})
    provider = registry.for_capability("herg")
    baseline = json.loads(GOLDEN.read_text())["models"]["herg-tox21-chemberta-v1"]["predictions"]
    case_ids = list(baseline)
    actual = provider.predict([baseline[c]["canonical_smiles"] for c in case_ids])
    return baseline, dict(zip(case_ids, actual)), provider


def test_baseline_covers_the_whole_panel(predictions):
    baseline, _, _ = predictions
    panel = json.loads((REPO / "benchmarks" / "fixtures" / "golden_panel.json").read_text())
    assert len(baseline) == panel["n_valid"]


def test_herg_probabilities_match_the_baseline(predictions):
    baseline, actual, _ = predictions
    for case_id, expected in baseline.items():
        assert actual[case_id]["herg_probability_blocker"] == pytest.approx(
            expected["herg_probability_blocker"], abs=TOLERANCE
        ), f"hERG drift on {case_id}"


def test_tox21_probabilities_match_the_baseline(predictions):
    baseline, actual, _ = predictions
    for case_id, expected in baseline.items():
        for task, value in expected["tox21_probability_activity"].items():
            assert actual[case_id]["tox21_probability_activity"][task] == pytest.approx(
                value, abs=TOLERANCE
            ), f"Tox21 drift on {case_id}/{task}"


def test_thresholds_come_from_the_artifact(predictions):
    _, _, provider = predictions
    assert provider.artifact_herg_threshold == pytest.approx(0.4133453071117401)
    thresholds = provider.artifact_tox21_thresholds
    assert len(thresholds) == 12
    assert thresholds["NR-AR"] == pytest.approx(0.9399998188018799)
    # The value the running service applied to every endpoint.
    assert provider.artifact_herg_threshold != 0.30


def test_identical_input_gives_identical_output(predictions):
    """Determinism within one process — the panel carries a duplicate case."""
    _, actual, _ = predictions
    assert actual["safe_acetaminophen"]["herg_probability_blocker"] == (
        actual["safe_paracetamol_dup"]["herg_probability_blocker"]
    )


def test_known_blockers_score_above_known_safe_drugs(predictions):
    """A coarse sanity check that survives a re-trained artifact."""
    _, actual, _ = predictions
    blockers = [v["herg_probability_blocker"] for k, v in actual.items() if k.startswith("herg_pos_")]
    safe = [v["herg_probability_blocker"] for k, v in actual.items() if k.startswith("safe_")]
    assert min(blockers) > max(safe), "hERG head no longer separates the control sets"


def test_service_starts_with_no_network(predictions):
    """The container sets HF_HUB_OFFLINE=1; the model must load anyway.

    Before the architecture config was vendored, this failed: the checkpoint
    carried every weight but the backbone was still constructed by resolving
    the model id against Hugging Face, so a fresh container could not reach
    readiness.
    """
    _, _, provider = predictions
    assert "offline" in provider.health().detail, provider.health().detail
    assert (REPO / "models" / "pretrained_2head_herg_chemberta_model"
            / "base_model" / "config.json").is_file()
