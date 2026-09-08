"""A minimal circuit breaker for an external evidence provider (remaining-plan
W3-06: "provider circuit breaker/backoff").

Without one, a provider having a bad minute gets hit with every single
``search_toxicology_evidence`` call anyone makes for as long as it stays
down — each one paying the full connect/read timeout before failing. This
trades that for: after enough consecutive failures, stop trying for a
cooldown window and fail fast instead, then let exactly one call through as
a trial once the cooldown elapses.

Pure state machine, no I/O of its own — the caller (``EuropePmcProvider``)
still owns the actual request and reports its own outcome back in with
``record_success``/``record_failure``. A ``clock`` callable (default
``time.monotonic``) is injectable so tests do not need to sleep for real.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Callable


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpen(Exception):
    """Raised by ``before_call`` while the circuit is open — the caller
    should map this to whatever typed error it normally raises for
    "provider unavailable", not let it leak as a bare exception."""


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        reset_after_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self._failure_threshold = failure_threshold
        self._reset_after_s = reset_after_s
        self._clock = clock
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        # True exactly while the one half-open trial call is in flight, so a
        # second caller arriving before it resolves does not also get let
        # through as a second "trial" — only one probe at a time.
        self._trial_in_flight = False

    @property
    def state(self) -> CircuitState:
        if self._trial_in_flight:
            return CircuitState.HALF_OPEN
        if self._opened_at is None:
            return CircuitState.CLOSED
        if self._clock() - self._opened_at >= self._reset_after_s:
            return CircuitState.HALF_OPEN
        return CircuitState.OPEN

    def before_call(self) -> None:
        """Raises ``CircuitOpen`` if this call should not even be attempted.
        A half-open circuit lets exactly one call through as a trial — a
        second caller arriving while that trial is still in flight is
        refused too, not treated as a second probe."""
        if self._trial_in_flight:
            raise CircuitOpen("a half-open trial call is already in flight")
        if self._opened_at is None:
            return
        if self._clock() - self._opened_at >= self._reset_after_s:
            self._trial_in_flight = True
            return
        raise CircuitOpen(
            f"circuit open after {self._consecutive_failures} consecutive failures; "
            f"retry after the {self._reset_after_s}s cooldown elapses"
        )

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self._trial_in_flight = False
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_at = self._clock()
