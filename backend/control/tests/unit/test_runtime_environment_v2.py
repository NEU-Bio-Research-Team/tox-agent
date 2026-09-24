import pytest

from toxagent.runtime.environment import runtime_environment


def test_runtime_environment_is_allowlist_not_inheritance():
    env = runtime_environment({
        "PATH": "/bin", "DATABASE_URL": "secret", "PREDICTOR_TOKEN": "secret",
        "RANDOM_PARENT_VALUE": "leak",
    }, additions={"TOXAGENT_RUN_ID": "run"})
    assert env == {"PATH": "/bin", "TOXAGENT_RUN_ID": "run"}


def test_sensitive_addition_is_refused():
    with pytest.raises(ValueError):
        runtime_environment({}, additions={"API_TOKEN": "secret"})
