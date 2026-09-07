"""ExplainService aggregation math (plan section 5.1). Stub attribution."""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")

from toxpred.application.explain import ExplainService  # noqa: E402


class _StubAttribution:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[tuple] = []

    def attribute(self, smiles, endpoint, task=None):
        self.calls.append((smiles, endpoint, task))
        return self.result


def _raw(tokens, *, status="completed", canonical="CCO"):
    return {
        "status": status,
        "input_smiles": canonical,
        "canonical_smiles": canonical,
        "endpoint": "herg",
        "task": None,
        "probability": 0.37,
        "tokens": tokens,
        "metadata": {
            "method": "grad_x_embedding_l2_v1",
            "model_id": "herg-tox21-chemberta-v1",
            "duration_ms": 12.3,
            "note": None,
        },
    }


def test_token_importances_are_projected_onto_atoms_and_normalised():
    tokens = [
        {"token": "<s>", "importance": 1.0, "offsets": [0, 0]},
        {"token": "C", "importance": 2.0, "offsets": [0, 1]},
        {"token": "C", "importance": 3.0, "offsets": [1, 2]},
        {"token": "O", "importance": 4.0, "offsets": [2, 3]},
        {"token": "</s>", "importance": 5.0, "offsets": [0, 0]},
    ]
    result = ExplainService(_StubAttribution(_raw(tokens))).explain("CCO", "herg")

    assert [a["atom_index"] for a in result["atoms"]] == [0, 1, 2]
    assert [a["symbol"] for a in result["atoms"]] == ["C", "C", "O"]
    assert [a["importance"] for a in result["atoms"]] == [2.0, 3.0, 4.0]
    # unmapped = 1.0 + 5.0 = 6.0; total = 2+3+4+6 = 15
    assert result["unmapped_importance"] == pytest.approx(6.0 / 15.0)
    rel_sum = sum(a["relative_importance"] for a in result["atoms"])
    assert rel_sum + result["unmapped_importance"] == pytest.approx(1.0)
    assert result["atom_order_version"] == "rdkit-output-order-v1"
    assert result["method"] == "grad_x_embedding_l2_v1+token_atom_align_v1"


def test_a_token_over_multiple_atoms_splits_its_importance_equally():
    tokens = [
        {"token": "CC", "importance": 6.0, "offsets": [0, 2]},
        {"token": "O", "importance": 3.0, "offsets": [2, 3]},
    ]
    result = ExplainService(_StubAttribution(_raw(tokens))).explain("CCO", "herg")
    importances = {a["atom_index"]: a["importance"] for a in result["atoms"]}
    assert importances == {0: 3.0, 1: 3.0, 2: 3.0}
    assert result["unmapped_importance"] == pytest.approx(0.0)


def test_partial_status_is_passed_through():
    tokens = [{"token": "C", "importance": 1.0, "offsets": [0, 1]}]
    result = ExplainService(
        _StubAttribution(_raw(tokens, status="partial"))
    ).explain("CCO", "herg")
    assert result["status"] == "partial"


def test_failed_attribution_yields_no_atoms_and_no_highlight():
    stub = _StubAttribution(
        {
            "status": "failed",
            "error": "RuntimeError",
            "message": "backward pass blew up",
            "input_smiles": "CCO",
            "canonical_smiles": "CCO",
            "endpoint": "herg",
            "task": None,
            "duration_ms": 5.0,
        }
    )
    result = ExplainService(stub).explain("CCO", "herg")
    assert result["status"] == "failed"
    assert result["atoms"] == []
    assert result["unmapped_importance"] is None
    assert result["probability"] is None
