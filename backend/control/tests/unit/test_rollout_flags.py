"""A rollout flag that outlives its rollout is a fork (ADR 0009).

These guards are cheap and they are the only thing standing between a
two-release compatibility window and a permanent second code path.
"""
from __future__ import annotations

import pytest

from toxagent.flags import FLAGS, MAX_FLAG_LIFETIME_DAYS, flag, is_enabled, rollout_matrix


def test_catalogue_is_not_empty_and_names_are_unique() -> None:
    names = [item.name for item in FLAGS]
    assert names, "the flag catalogue must list the rollout controls that exist"
    assert len(names) == len(set(names))


@pytest.mark.parametrize("item", FLAGS, ids=lambda item: item.name)
def test_every_flag_has_an_owner_and_a_removal_plan(item) -> None:
    assert item.owner, f"{item.name} has no owner"
    assert item.removal_condition, f"{item.name} does not say what makes it deletable"
    assert item.remove_by > item.added_on, f"{item.name} expires before it was added"
    lifetime = (item.remove_by - item.added_on).days
    assert lifetime <= MAX_FLAG_LIFETIME_DAYS, (
        f"{item.name} is scheduled to live {lifetime} days; the plan allows "
        f"{MAX_FLAG_LIFETIME_DAYS} (two releases)"
    )


@pytest.mark.parametrize("item", FLAGS, ids=lambda item: item.name)
def test_env_var_is_derived_from_the_name(item) -> None:
    assert item.env_var == f"TOXAGENT_FLAG_{item.name.upper()}"


def test_environment_overrides_the_default(monkeypatch) -> None:
    target = FLAGS[0]
    monkeypatch.setenv(target.env_var, "0" if target.default else "1")
    assert is_enabled(target.name) is (not target.default)


def test_explicit_overrides_beat_the_environment(monkeypatch) -> None:
    target = FLAGS[0]
    monkeypatch.setenv(target.env_var, "1")
    assert is_enabled(target.name, {target.name: False}) is False


def test_unknown_flag_is_a_typed_error() -> None:
    with pytest.raises(KeyError, match="unknown rollout flag"):
        flag("no_such_flag")


def test_rollout_matrix_reports_every_flag() -> None:
    matrix = rollout_matrix()
    assert set(matrix) == {item.name for item in FLAGS}
    for row in matrix.values():
        assert set(row) == {
            "env_var",
            "default",
            "owner",
            "added_on",
            "remove_by",
            "removal_condition",
        }
