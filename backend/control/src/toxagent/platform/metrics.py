"""Low-cardinality product metrics, exposed in Prometheus text format (WS12 / PR-18).

The remediation plan's rule for metrics is short: no SMILES, no prose, no URL,
no owner, session or run id in a label. Those belong in logs under access
control. A label that carries one turns every run into its own time series,
which is both a privacy leak and the fastest way to take down a metrics store.

So the rule is enforced here rather than trusted to every call site:

* every metric is declared once, below, with its label keys; ``inc`` or
  ``observe`` on an undeclared name, or with undeclared label keys, raises —
  a programming error the unit suite catches;
* a label *value* that looks like a product id, a URL or a long opaque token is
  replaced with ``invalid`` and counted, rather than raising, because a metrics
  call must never be the thing that fails a user's run;
* each metric holds at most ``MAX_SERIES`` label combinations; past that, new
  combinations fold into one ``overflow`` series.

In-process and per-replica on purpose. Aggregation across replicas is the
scraper's job; sharing counters through the database would put a write on
every run for a number nobody reads per request.

Every name declared here is documented in ``docs/observability/METRICS.md``, and
``tests/unit/test_metrics.py`` fails when the two disagree.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

#: The only label keys any metric may declare. Adding one is a review decision.
ALLOWED_LABEL_KEYS: frozenset[str] = frozenset(
    {
        "intent", "queue", "outcome", "result", "failure_code", "stage", "status",
        "profile", "attempt", "scope", "reason", "endpoint", "violation_code",
    }
)

#: Product ids (``run_…``, ``ses_…`` and friends), URLs, long opaque tokens,
#: and anything with whitespace or quoting in it (prose, SMILES with brackets).
_FORBIDDEN_VALUE = re.compile(
    r"^[a-z]{2,4}_[0-9a-f]{8,}$|://|[A-Za-z0-9+/=_-]{33,}|[\s{}\"\\\[\]()=#@]"
)

MAX_SERIES = 200
MAX_VALUE_LENGTH = 64

DEFAULT_BUCKETS: tuple[float, ...] = (
    0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 15, 30, 60, 120, 300, 600,
)


class _Rejections:
    def __init__(self) -> None:
        self.value = 0
        self._lock = threading.Lock()

    def increment(self) -> None:
        with self._lock:
            self.value += 1


REJECTIONS = _Rejections()


def _clean(value: object) -> str:
    text = "" if value is None else str(value)
    if len(text) > MAX_VALUE_LENGTH or _FORBIDDEN_VALUE.search(text):
        REJECTIONS.increment()
        return "invalid"
    return text


@dataclass
class Metric:
    name: str
    help: str
    kind: str  # counter | histogram
    labels: tuple[str, ...]
    buckets: tuple[float, ...] = ()
    _series: dict[tuple[str, ...], list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _key(self, labels: Mapping[str, object]) -> tuple[str, ...]:
        if set(labels) != set(self.labels):
            raise ValueError(
                f"metric {self.name} takes labels {sorted(self.labels)}, got {sorted(labels)}"
            )
        key = tuple(_clean(labels[name]) for name in self.labels)
        if key not in self._series and len(self._series) >= MAX_SERIES:
            key = tuple("overflow" for _ in self.labels)
        return key

    def _cell(self, key: tuple[str, ...]) -> list[float]:
        cell = self._series.get(key)
        if cell is None:
            cell = [0.0] * (len(self.buckets) + 2 if self.kind == "histogram" else 1)
            self._series[key] = cell
        return cell

    def series(self) -> dict[tuple[str, ...], list[float]]:
        with self._lock:
            return {key: list(cell) for key, cell in self._series.items()}


class Registry:
    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def declare(
        self, name: str, help: str, *, kind: str, labels: Sequence[str] = (),
        buckets: Iterable[float] = DEFAULT_BUCKETS,
    ) -> Metric:
        if not re.fullmatch(r"toxagent_[a-z0-9_]+", name):
            raise ValueError(f"metric name {name!r} must be toxagent_<snake_case>")
        if kind == "counter" and not name.endswith("_total"):
            raise ValueError(f"counter {name} must end in _total")
        if kind == "histogram" and not name.endswith("_seconds"):
            raise ValueError(f"histogram {name} must be measured in seconds")
        bad = set(labels) - ALLOWED_LABEL_KEYS
        if bad:
            raise ValueError(f"metric {name} uses label keys outside the allowlist: {sorted(bad)}")
        if name in self._metrics:
            raise ValueError(f"metric {name} is already declared")
        metric = Metric(
            name, help, kind, tuple(labels),
            tuple(sorted(buckets)) if kind == "histogram" else (),
        )
        self._metrics[name] = metric
        return metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError:
            raise KeyError(f"metric {name!r} is not declared in toxagent.platform.metrics") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._metrics))

    def reset(self) -> None:
        """For tests: forget every observation, keep every declaration."""
        for metric in self._metrics.values():
            with metric._lock:
                metric._series.clear()
        REJECTIONS.value = 0

    def render(self) -> str:
        lines: list[str] = []
        for name in self.names():
            metric = self._metrics[name]
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} {metric.kind}")
            for key, cell in sorted(metric.series().items()):
                base = dict(zip(metric.labels, key))
                if metric.kind == "histogram":
                    for bound, count in zip(metric.buckets, cell):
                        lines.append(f"{name}_bucket{_labels({**base, 'le': _num(bound)})} {_num(count)}")
                    lines.append(f"{name}_bucket{_labels({**base, 'le': '+Inf'})} {_num(cell[-1])}")
                    lines.append(f"{name}_sum{_labels(base)} {_num(cell[-2])}")
                    lines.append(f"{name}_count{_labels(base)} {_num(cell[-1])}")
                else:
                    lines.append(f"{name}{_labels(base)} {_num(cell[0])}")
        lines.append(
            "# HELP toxagent_metrics_label_rejections_total Label values replaced "
            "because they looked like an id, URL, token or prose."
        )
        lines.append("# TYPE toxagent_metrics_label_rejections_total counter")
        lines.append(f"toxagent_metrics_label_rejections_total {REJECTIONS.value}")
        return "\n".join(lines) + "\n"


def _labels(labels: Mapping[str, str]) -> str:
    if not labels:
        return ""
    return "{" + ",".join(f'{key}="{value}"' for key, value in labels.items()) + "}"


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


REGISTRY = Registry()


def inc(name: str, amount: float = 1.0, **labels: object) -> None:
    metric = REGISTRY.get(name)
    if metric.kind != "counter":
        raise TypeError(f"{name} is a {metric.kind}, not a counter")
    if amount < 0:
        raise ValueError("a counter only goes up")
    key = metric._key(labels)
    with metric._lock:
        metric._cell(key)[0] += amount


def observe(name: str, value: float, **labels: object) -> None:
    metric = REGISTRY.get(name)
    if metric.kind != "histogram":
        raise TypeError(f"{name} is a {metric.kind}, not a histogram")
    key = metric._key(labels)
    if value is None or math.isnan(value) or value < 0:
        return
    with metric._lock:
        cell = metric._cell(key)
        for index, bound in enumerate(metric.buckets):
            if value <= bound:
                cell[index] += 1
        cell[-2] += value
        cell[-1] += 1


# --- the dictionary ---------------------------------------------------------

REGISTRY.declare(
    "toxagent_run_claims_total",
    "Claim decisions a worker made on a job: claimed, deferred, cancelled_before_execution, recovery_exhausted.",
    kind="counter", labels=("queue", "result"),
)
REGISTRY.declare(
    "toxagent_concurrency_refusals_total",
    "Claims refused a concurrency slot, by the scope whose cap was full.",
    kind="counter", labels=("scope",),
)
REGISTRY.declare(
    "toxagent_runs_finished_total",
    "Runs whose execution ended on this worker, by intent, queue and outcome.",
    kind="counter", labels=("intent", "queue", "outcome"),
)
REGISTRY.declare(
    "toxagent_run_duration_seconds",
    "Wall time a worker spent executing a run, by queue and outcome.",
    kind="histogram", labels=("queue", "outcome"),
)
REGISTRY.declare(
    "toxagent_run_queue_wait_seconds",
    "Time from a job being written to its first claim.",
    kind="histogram", labels=("queue",),
)
REGISTRY.declare(
    "toxagent_report_stage_seconds",
    "Wall time of one report stage attempt, by stage and how it settled.",
    kind="histogram", labels=("stage", "status"),
)
REGISTRY.declare(
    "toxagent_report_synthesis_submissions_total",
    "Synthesis submissions judged, by attempt number and outcome.",
    kind="counter", labels=("attempt", "outcome"),
)
REGISTRY.declare(
    "toxagent_report_synthesis_violations_total",
    "Typed violations in refused synthesis submissions.",
    kind="counter", labels=("violation_code",),
)
REGISTRY.declare(
    "toxagent_worker_handoffs_total",
    "Runs a worker gave back to the queue instead of finishing, by reason.",
    kind="counter", labels=("reason",),
)
