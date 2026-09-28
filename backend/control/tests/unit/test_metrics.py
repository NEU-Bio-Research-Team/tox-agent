"""The metric registry keeps ids and prose out of labels (PR-18).

A metric label that carries a run id or a molecule is a privacy leak and a
cardinality explosion at once. These pin the guards, and pin the dictionary in
docs/reference/metrics.md to what the code declares.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from toxagent.platform import metrics

DOC = Path(__file__).resolve().parents[4] / "docs" / "reference" / "metrics.md"


@pytest.fixture(autouse=True)
def _clean_registry():
    metrics.REGISTRY.reset()
    yield
    metrics.REGISTRY.reset()


def test_every_declared_metric_is_documented_and_nothing_else_is():
    documented = set(re.findall(r"^\| `(toxagent_[a-z0-9_]+)` \|", DOC.read_text(), re.M))
    assert documented == set(metrics.REGISTRY.names())


def test_documented_labels_match_the_declaration():
    rows = re.findall(r"^\| `(toxagent_[a-z0-9_]+)` \| (\w+) \| ([^|]*) \|", DOC.read_text(), re.M)
    for name, kind, labels in rows:
        metric = metrics.REGISTRY.get(name)
        assert metric.kind == kind, name
        assert set(re.findall(r"`(\w+)`", labels)) == set(metric.labels), name


@pytest.mark.parametrize(
    "value",
    [
        "run_" + "a" * 32,
        "ses_0123456789abcdef",
        "https://example.org/paper",
        "CC(=O)Oc1ccccc1C(=O)O",
        "the model says this is safe",
        "x" * 80,
    ],
)
def test_an_id_url_molecule_or_prose_never_becomes_a_label(value):
    metrics.inc("toxagent_run_claims_total", queue=value, result="claimed")
    rendered = metrics.REGISTRY.render()
    assert value not in rendered
    assert 'queue="invalid"' in rendered
    assert "toxagent_metrics_label_rejections_total 1" in rendered


def test_an_undeclared_metric_or_label_is_a_programming_error():
    with pytest.raises(KeyError):
        metrics.inc("toxagent_not_declared_total")
    with pytest.raises(ValueError):
        metrics.inc("toxagent_run_claims_total", queue="report")
    with pytest.raises(ValueError):
        metrics.inc("toxagent_run_claims_total", queue="report", result="claimed", run_id="x")
    with pytest.raises(TypeError):
        metrics.observe("toxagent_run_claims_total", 1.0, queue="report", result="claimed")


def test_series_past_the_cap_fold_into_overflow():
    for index in range(metrics.MAX_SERIES + 25):
        metrics.inc("toxagent_concurrency_refusals_total", scope=f"s{index}")
    series = metrics.REGISTRY.get("toxagent_concurrency_refusals_total").series()
    assert len(series) == metrics.MAX_SERIES + 1
    assert series[("overflow",)] == [25.0]


def test_prometheus_text_for_a_histogram():
    metrics.observe("toxagent_run_queue_wait_seconds", 0.3, queue="report")
    metrics.observe("toxagent_run_queue_wait_seconds", 7, queue="report")
    text = metrics.REGISTRY.render()
    assert "# TYPE toxagent_run_queue_wait_seconds histogram" in text
    assert 'toxagent_run_queue_wait_seconds_bucket{queue="report",le="0.5"} 1' in text
    assert 'toxagent_run_queue_wait_seconds_bucket{queue="report",le="+Inf"} 2' in text
    assert 'toxagent_run_queue_wait_seconds_count{queue="report"} 2' in text
    assert 'toxagent_run_queue_wait_seconds_sum{queue="report"} 7.3' in text


def test_declarations_refuse_names_and_labels_outside_the_rules():
    registry = metrics.Registry()
    with pytest.raises(ValueError):
        registry.declare("runs_total", "x", kind="counter")
    with pytest.raises(ValueError):
        registry.declare("toxagent_runs", "x", kind="counter")
    with pytest.raises(ValueError):
        registry.declare("toxagent_x_total", "x", kind="counter", labels=("session_id",))
