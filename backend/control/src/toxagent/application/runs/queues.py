"""Which kind of worker a run belongs to (WS08 / PR-14).

Three classes, chosen by what the work costs rather than by who asked:

* **interactive** — a question with a person waiting on the answer. Seconds.
* **report** — a whole report build. Minutes, and a model turn at the end.
* **deterministic** — predictor work: analyses, batches, structure recognition.
  No model, bounded by the predictor's own latency.

They are separate so they can be scaled and capped separately. A deployment
that ran reports and questions from one pool would let three concurrent report
builds occupy every slot while a user's one-line question waited behind them.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from ...domain.run import Intent


class QueueClass(str, Enum):
    INTERACTIVE = "interactive"
    REPORT = "report"
    DETERMINISTIC = "deterministic"


ALL_QUEUES: tuple[str, ...] = tuple(queue.value for queue in QueueClass)

_BY_INTENT: Mapping[Intent, QueueClass] = {
    Intent.BUILD_REPORT: QueueClass.REPORT,
    Intent.ANALYSIS: QueueClass.DETERMINISTIC,
    Intent.ANALYSIS_BATCH: QueueClass.DETERMINISTIC,
    Intent.STRUCTURE_RECOGNITION: QueueClass.DETERMINISTIC,
}

#: Claim order when one worker serves several queues. Lower first: a person
#: waiting on an answer outranks a build nobody is watching second by second.
PRIORITY: Mapping[QueueClass, int] = {
    QueueClass.INTERACTIVE: 0,
    QueueClass.DETERMINISTIC: 10,
    QueueClass.REPORT: 20,
}


def queue_for_intent(intent: Intent | str) -> QueueClass:
    try:
        resolved = intent if isinstance(intent, Intent) else Intent(intent)
    except ValueError:
        return QueueClass.INTERACTIVE
    return _BY_INTENT.get(resolved, QueueClass.INTERACTIVE)


def queue_of_job(job: Mapping[str, Any]) -> str:
    """The job's queue, derived from its envelope when the row predates 0014."""
    stored = job.get("queue_name")
    if stored:
        return str(stored)
    envelope = job.get("envelope") or {}
    return queue_for_intent(str(envelope.get("intent") or "")).value


def parse_queues(names: tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(names) - set(ALL_QUEUES))
    if unknown:
        raise ValueError(f"unknown worker queue(s) {unknown}; known: {list(ALL_QUEUES)}")
    if not names:
        raise ValueError("a worker must serve at least one queue")
    return names
