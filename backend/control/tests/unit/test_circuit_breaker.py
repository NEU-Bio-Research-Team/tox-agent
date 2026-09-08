"""CircuitBreaker (remaining-plan W3-06). Pure state machine — a fake clock,
never a real sleep, so these run in milliseconds."""
from __future__ import annotations

import pytest

from toxagent.research.circuit_breaker import CircuitBreaker, CircuitOpen, CircuitState


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_starts_closed():
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=FakeClock())
    assert breaker.state is CircuitState.CLOSED
    breaker.before_call()  # must not raise


def test_stays_closed_below_the_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=FakeClock())
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.before_call()  # must not raise


def test_opens_after_reaching_the_failure_threshold():
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=FakeClock())
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpen):
        breaker.before_call()


def test_a_success_resets_the_consecutive_failure_count():
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=FakeClock())
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    # Two more failures after the reset is still below threshold (3), not
    # four cumulative — the whole point of "consecutive".
    assert breaker.state is CircuitState.CLOSED


def test_half_opens_after_the_cooldown_and_allows_exactly_one_trial():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    clock.advance(10)
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.before_call()  # the one trial call is allowed through
    # A second caller arriving while the trial is still in flight must not
    # also be let through as a second probe.
    with pytest.raises(CircuitOpen):
        breaker.before_call()


def test_a_successful_trial_closes_the_circuit():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(10)
    breaker.before_call()
    breaker.record_success()

    assert breaker.state is CircuitState.CLOSED
    breaker.before_call()  # must not raise


def test_a_failed_trial_reopens_the_circuit_with_a_fresh_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, reset_after_s=10, clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(10)
    breaker.before_call()
    breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitOpen):
        breaker.before_call()

    # The fresh cooldown starts from the failed trial, not the original open.
    clock.advance(9)
    assert breaker.state is CircuitState.OPEN
    clock.advance(1)
    assert breaker.state is CircuitState.HALF_OPEN
