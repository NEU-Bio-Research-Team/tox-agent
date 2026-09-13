"""Turn provider usage reports into facts that can be added up.

P1-1 of the 2026-09-13 audit: one Q&A run produced 21 usage events, in which
the snapshot ``input=3826 / output=101 / reasoning=24`` appeared three times,
interleaved with lifecycle events carrying nothing but zeroes. Every one of
them was stored as an independent row with no source identity, so any dashboard
that summed the column reported roughly three times the tokens the run actually
used.

The fix is not deduplication after the fact. It is admitting that a provider
report is *evidence of a state*, not an increment:

**Cumulative** — the report restates the running total for something (an
assistant message, a turn). Two reports of the same thing are the same fact
seen twice; the later revision replaces the earlier one, and adding them is
always wrong.

**Delta** — the report states what was consumed since the last one. These add,
and each one must be counted exactly once, which is what the source identity is
for.

**Unknown** — a provider this module has no contract with. Rows stay readable
and stay out of every total, because a total that silently includes guesses is
worse than one that says it is incomplete.

For OpenCode V1 specifically: ``message.part.updated`` with a ``step-finish``
part and ``message.updated`` both restate the same assistant message's running
total. This module keeps one fact per *distinct running total per message* —
a restatement of numbers already held establishes nothing — and drops the
zero-only lifecycle report, which carries no information a later report does
not.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

#: Field names this module reads out of a normalized runtime event payload.
#: The adapter is responsible for filling them; anything it cannot determine
#: stays absent rather than being invented.
SOURCE_EVENT_TYPE = "source_event_type"
PROVIDER_MESSAGE_ID = "provider_message_id"
PROVIDER_STEP_ID = "provider_step_id"
REVISION = "revision"


class UsageSemantics(str, Enum):
    CUMULATIVE = "cumulative"
    DELTA = "delta"
    UNKNOWN = "unknown"


_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "total_tokens",
)


@dataclass(frozen=True, slots=True)
class UsageCounts:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def is_all_zero(self) -> bool:
        """True when every reported field is zero and none is missing-but-set.

        A provider's "the step began" lifecycle event looks exactly like this.
        It is not a measurement of zero usage; it is the absence of one.
        """
        values = [getattr(self, name) for name in _TOKEN_FIELDS]
        reported = [value for value in values if value is not None]
        return bool(reported) and all(value == 0 for value in reported)

    @property
    def is_empty(self) -> bool:
        return all(getattr(self, name) is None for name in _TOKEN_FIELDS)


@dataclass(frozen=True, slots=True)
class NormalizedUsage:
    """One usage fact, addressable by its source so it cannot be stored twice."""

    #: Stable identity of the provider report this came from. Derived when the
    #: provider does not supply one, but always a pure function of the report's
    #: own identifying fields — never of arrival order or wall-clock time.
    source_event_id: str
    source_event_type: str
    semantics: UsageSemantics
    counts: UsageCounts
    provider_message_id: str | None = None
    provider_step_id: str | None = None
    revision: int | None = None
    raw_payload_hash: str = ""

    #: The scope a cumulative snapshot is cumulative *over*. Two snapshots
    #: sharing this key are the same running total seen twice.
    @property
    def scope_key(self) -> str:
        return self.provider_message_id or self.provider_step_id or self.source_event_id


def _counts_from_payload(payload: Mapping[str, Any]) -> UsageCounts:
    tokens = payload.get("tokens")
    tokens = tokens if isinstance(tokens, Mapping) else {}
    cache = tokens.get("cache")
    cache = cache if isinstance(cache, Mapping) else {}

    def count(*candidates: Any) -> int | None:
        for value in candidates:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return None

    return UsageCounts(
        input_tokens=count(tokens.get("input"), tokens.get("input_tokens")),
        output_tokens=count(tokens.get("output"), tokens.get("output_tokens")),
        reasoning_tokens=count(tokens.get("reasoning"), tokens.get("reasoning_tokens")),
        cache_read_tokens=count(cache.get("read"), tokens.get("cache_read_tokens")),
        cache_write_tokens=count(cache.get("write"), tokens.get("cache_write_tokens")),
        total_tokens=count(tokens.get("total"), tokens.get("total_tokens")),
    )


def payload_hash(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]


def cumulative_source_event_id(
    *, runtime_session_id: str, scope_key: str, counts: UsageCounts
) -> str:
    """The identity of one *running total*, independent of how it arrived.

    OpenCode V1 restates the same assistant message's total on both a
    ``step-finish`` part and a ``message.updated`` envelope. Those are one fact
    reported twice, so their identity must not include the event type — if it
    did, the audit's three copies would become two rows instead of one, and the
    database index could not catch the third after a worker restart.

    The counts are part of the identity because that is what a cumulative
    report *is*: a statement that this scope now stands at these numbers. A
    later report with different numbers is a new fact; one with the same
    numbers is the same fact again.
    """
    material = "|".join(
        str(part)
        for part in (
            runtime_session_id,
            "cumulative",
            scope_key,
            *[getattr(counts, name) for name in _TOKEN_FIELDS],
        )
    )
    return "usg_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def derive_source_event_id(
    *,
    runtime_session_id: str,
    source_event_type: str,
    provider_message_id: str | None,
    provider_step_id: str | None,
    revision: int | None,
    counts: UsageCounts,
) -> str:
    """A stable id for a provider report that did not carry one.

    Built from what identifies the *report*, including the counts: OpenCode V1
    does not version ``step-finish`` parts, so two genuinely different steps of
    the same message are told apart by what they measured. Arrival order is
    deliberately not an input — a reconnect that replays the stream must derive
    the same ids, or recovery would double every total.
    """
    material = "|".join(
        str(part)
        for part in (
            runtime_session_id,
            source_event_type,
            provider_message_id or "",
            provider_step_id or "",
            revision if revision is not None else "",
            *[getattr(counts, name) for name in _TOKEN_FIELDS],
        )
    )
    return "usg_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class RuntimeUsageNormalizer:
    """Per runtime session. Not thread-safe; one run, one instance.

    The in-memory state here is a fast path, not the guarantee. The database's
    unique index on ``(runtime_binding_id, source_event_id)`` is what survives
    a worker restart mid-run, and this class is written so that re-feeding
    every event after a reconnect produces exactly the same set of facts.
    """

    #: Which OpenCode V1 event types restate a running total.
    _CUMULATIVE_TYPES = frozenset(
        {"message.updated", "message.part.updated:step-finish"}
    )

    def __init__(self, runtime_session_id: str, *, provider: str = "opencode") -> None:
        self._runtime_session_id = runtime_session_id
        self._provider = provider
        #: source_event_id -> the fact already emitted for it.
        self._seen: dict[str, NormalizedUsage] = {}
        #: scope_key -> the latest fact accepted for that scope, so a
        #: restatement is recognised and an out-of-order older revision is
        #: ignored rather than overwriting a newer total.
        self._latest: dict[str, NormalizedUsage] = {}

    def semantics_for(self, source_event_type: str) -> UsageSemantics:
        if self._provider != "opencode":
            return UsageSemantics.UNKNOWN
        if source_event_type in self._CUMULATIVE_TYPES:
            return UsageSemantics.CUMULATIVE
        return UsageSemantics.UNKNOWN

    def accept(self, payload: Mapping[str, Any]) -> NormalizedUsage | None:
        """Return the fact this report establishes, or ``None``.

        ``None`` means one of three things, all of which are correct outcomes
        and none of which is an error: the report carried no counts, it carried
        only zeroes (a lifecycle marker), or it restates a fact already held.
        """
        counts = _counts_from_payload(payload)
        if counts.is_empty or counts.is_all_zero:
            return None

        source_event_type = str(payload.get(SOURCE_EVENT_TYPE) or "unknown")
        message_id = payload.get(PROVIDER_MESSAGE_ID)
        step_id = payload.get(PROVIDER_STEP_ID)
        raw_revision = payload.get(REVISION)
        revision = (
            raw_revision
            if isinstance(raw_revision, int) and not isinstance(raw_revision, bool)
            else None
        )
        semantics = self.semantics_for(source_event_type)
        message_id = message_id if isinstance(message_id, str) else None
        step_id = step_id if isinstance(step_id, str) else None

        supplied = payload.get("source_event_id")
        if semantics is UsageSemantics.CUMULATIVE:
            scope_key = message_id or step_id or source_event_type
            held = self._latest.get(scope_key)
            if held is not None:
                if held.counts == counts:
                    # The same running total, reported again — on the other
                    # envelope type, or simply twice. This is P1-1.
                    return None
                if (
                    revision is not None
                    and held.revision is not None
                    and revision < held.revision
                ):
                    # An older revision arriving late says nothing new about a
                    # scope we already hold a later total for.
                    return None
            source_event_id = supplied or cumulative_source_event_id(
                runtime_session_id=self._runtime_session_id,
                scope_key=scope_key,
                counts=counts,
            )
        else:
            source_event_id = supplied or derive_source_event_id(
                runtime_session_id=self._runtime_session_id,
                source_event_type=source_event_type,
                provider_message_id=message_id,
                provider_step_id=step_id,
                revision=revision,
                counts=counts,
            )
        if source_event_id in self._seen:
            return None

        fact = NormalizedUsage(
            source_event_id=str(source_event_id),
            source_event_type=source_event_type,
            semantics=semantics,
            counts=counts,
            provider_message_id=message_id,
            provider_step_id=step_id,
            revision=revision,
            raw_payload_hash=payload_hash(payload),
        )
        if semantics is UsageSemantics.CUMULATIVE:
            self._latest[fact.scope_key] = fact
        self._seen[fact.source_event_id] = fact
        return fact


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """The authoritative aggregate. Says how it was computed, and how sure."""

    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    total_tokens: int | None
    #: ``latest_per_scope`` | ``sum_of_deltas`` | ``mixed`` | ``none``
    aggregation_method: str
    #: ``complete`` | ``partial`` | ``unknown``
    completeness: str
    fact_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "reasoning": self.reasoning_tokens,
                "cache_read": self.cache_read_tokens,
                "cache_write": self.cache_write_tokens,
                "total": self.total_tokens,
            },
            "aggregation_method": self.aggregation_method,
            "completeness": self.completeness,
            "fact_count": self.fact_count,
        }


def summarize(facts: Iterable[NormalizedUsage]) -> UsageSummary:
    """Aggregate normalized facts. Never adds two snapshots of one scope.

    A cumulative scope contributes its highest revision once. Deltas add. A
    run that mixed semantics, or that contains anything ``unknown``, is
    reported as ``partial``/``unknown`` rather than given a number a reader
    would take for a measurement.
    """
    facts = list(facts)
    if not facts:
        return UsageSummary(None, None, None, None, None, None, "none", "unknown", 0)

    latest_by_scope: dict[str, NormalizedUsage] = {}
    deltas: list[NormalizedUsage] = []
    unknown = 0
    for fact in facts:
        if fact.semantics is UsageSemantics.CUMULATIVE:
            current = latest_by_scope.get(fact.scope_key)
            if current is None or (fact.revision or 0) >= (current.revision or 0):
                latest_by_scope[fact.scope_key] = fact
        elif fact.semantics is UsageSemantics.DELTA:
            deltas.append(fact)
        else:
            unknown += 1

    contributing = list(latest_by_scope.values()) + deltas
    if not contributing:
        return UsageSummary(None, None, None, None, None, None, "none", "unknown", len(facts))

    totals: dict[str, int | None] = {}
    for name in _TOKEN_FIELDS:
        values = [getattr(fact.counts, name) for fact in contributing]
        reported = [value for value in values if value is not None]
        totals[name] = sum(reported) if reported else None

    if latest_by_scope and deltas:
        method = "mixed"
    elif deltas:
        method = "sum_of_deltas"
    else:
        method = "latest_per_scope"

    if unknown:
        completeness = "partial"
    elif any(totals[name] is None for name in ("input_tokens", "output_tokens")):
        completeness = "partial"
    else:
        completeness = "complete"

    return UsageSummary(
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        reasoning_tokens=totals["reasoning_tokens"],
        cache_read_tokens=totals["cache_read_tokens"],
        cache_write_tokens=totals["cache_write_tokens"],
        total_tokens=totals["total_tokens"],
        aggregation_method=method,
        completeness=completeness,
        fact_count=len(facts),
    )
