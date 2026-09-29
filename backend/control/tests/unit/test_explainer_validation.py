"""The explainer statement matches the benchmark it claims to copy (RETHINK §2, §4.3)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from toxagent.domain import explainer_validation as ev

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("item", ev.MEASUREMENTS, ids=lambda item: item.source.rsplit("/", 1)[-1])
def test_each_measurement_is_the_result_file_it_names(item):
    summary = json.loads((REPO_ROOT / item.source).read_text())["summary"]
    assert (summary["model_id"], summary["endpoint"], summary["task"]) == (
        item.model_id, item.endpoint, item.task
    )
    assert summary["molecules_attempted"] == item.molecules_attempted
    assert summary["determinism_checked"] == item.determinism_checked
    assert len(summary["determinism_failures"]) == item.determinism_failures
    assert summary["spelling_checked"] == item.spelling_checked
    assert len(summary["spelling_failures"]) == item.spelling_failures
    assert summary["faithfulness_checked"] == item.faithfulness_checked
    assert summary["faithfulness_wins"] == item.faithfulness_wins
    assert summary["median_top_k_delta"] == item.median_top_k_delta
    assert summary["median_random_delta"] == item.median_random_delta


def test_every_result_file_on_disk_is_represented():
    results = sorted((REPO_ROOT / "backend/predictor/evals/xai/results").glob("*.json"))
    assert {f"backend/predictor/evals/xai/results/{p.name}" for p in results} == {
        item.source for item in ev.MEASUREMENTS
    }


def test_the_measured_targets_are_not_better_than_the_random_control():
    for item in ev.MEASUREMENTS:
        assert item.faithfulness == ev.Faithfulness.NOT_BETTER_THAN_RANDOM


@pytest.mark.parametrize(
    "method", ["grad_x_input_v2", "grad_x_input_v2+token_structure_align_v2", None],
)
def test_the_served_method_resolves_whichever_alignment_projected_it(method):
    view = ev.model_view("herg-tox21-chemberta-v1", "herg", None, method)
    assert view["faithfulness_vs_random_control"] == ev.Faithfulness.NOT_BETTER_THAN_RANDOM
    assert view["determinism"] == "passed"


@pytest.mark.parametrize(
    "target",
    [
        ("herg-tox21-chemberta-v1", "tox21", "SR-p53", "grad_x_input_v2"),
        ("herg-tox21-chemberta-v1", "herg", None, "integrated_gradients_v1"),
        ("some-other-model", "herg", None, "grad_x_input_v2"),
        (None, None, None, None),
    ],
)
def test_an_unmeasured_target_says_so_rather_than_borrowing_a_measurement(target):
    view = ev.model_view(*target)
    assert view["faithfulness_vs_random_control"] == ev.Faithfulness.NOT_MEASURED
    assert "never as mechanism" in view["reading"]


def test_the_model_view_carries_no_count_a_model_could_not_cite():
    for item in ev.MEASUREMENTS:
        view = ev.model_view(item.model_id, item.endpoint, item.task, item.method)
        assert "counts" not in view
        text = json.dumps({k: v for k, v in view.items() if k != "measured_on"})
        assert not re.search(r"\d", text.replace(ev.SCHEMA_VERSION, "")), text


def test_the_ui_view_adds_the_counts_and_the_source():
    view = ev.ui_view("herg-tox21-chemberta-v1", "herg", None, "grad_x_input_v2")
    assert view["counts"]["faithfulness_wins"] == "15/34"
    assert view["source"].endswith("herg-20260909.json")
