"""The Tox21 task order is positional and must never drift."""
import pytest

from toxpred.domain.endpoints import (
    TOX21_TASK_INDEX,
    TOX21_TASKS,
    Endpoint,
    validate_task_order,
)


def test_twelve_tasks_in_checkpoint_order():
    assert len(TOX21_TASKS) == 12
    assert TOX21_TASKS[0] == "NR-AR"
    assert TOX21_TASKS[-1] == "SR-p53"
    assert len(set(TOX21_TASKS)) == 12


def test_index_matches_tuple_positions():
    for position, task in enumerate(TOX21_TASKS):
        assert TOX21_TASK_INDEX[task] == position


def test_validate_accepts_the_frozen_order():
    validate_task_order(list(TOX21_TASKS))


def test_validate_rejects_a_permutation():
    permuted = list(TOX21_TASKS)
    permuted[0], permuted[1] = permuted[1], permuted[0]
    with pytest.raises(ValueError, match="task order mismatch"):
        validate_task_order(permuted)


def test_validate_rejects_a_truncated_order():
    with pytest.raises(ValueError):
        validate_task_order(list(TOX21_TASKS[:11]))


def test_endpoints_are_disjoint_concepts():
    assert {e.value for e in Endpoint} == {"clintox", "herg", "tox21"}
