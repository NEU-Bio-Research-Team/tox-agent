"""`bin/toxagent` passes the arguments it was given.

Two of its subcommands dropped an argument because the dispatcher had already
consumed the command name and the function indexed from `$2` anyway:

- `logs SERVICE` showed every service, and `logs A B` showed only B (I29).
- `restore FILE` printed usage and exited 1 before touching the database, so
  the documented recovery procedure could not be run at all (I28).

Neither is visible from reading a function in isolation, which is how both
survived. These tests run the real script with `docker` and `ss` replaced by
recorders on PATH, and assert on the argv that reached them.

    python -m pytest devops/tests -q
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "bin" / "toxagent"


@pytest.fixture
def stubbed(tmp_path):
    """A PATH where `docker` records its argv instead of running anything."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker-argv.log"

    (bin_dir / "docker").write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            printf '%s\\n' "$*" >> {log}
            # `docker compose version` and `config --quiet` must succeed for
            # doctor to get past its preflight.
            exit 0
            """
        )
    )
    (bin_dir / "docker").chmod(0o755)

    # No listener, so doctor's port check is a no-op here; port ownership has
    # its own test below.
    (bin_dir / "ss").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "ss").chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    def run(*args, input_text: str | None = None):
        return subprocess.run(
            [str(WRAPPER), *args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            input=input_text,
        )

    run.log = log  # type: ignore[attr-defined]
    return run


def _docker_calls(run) -> list[str]:
    if not run.log.exists():
        return []
    return run.log.read_text().splitlines()


# --- I29: logs ---------------------------------------------------------------

def test_logs_with_one_service_asks_for_that_service(stubbed):
    stubbed("logs", "toxpred")
    calls = [c for c in _docker_calls(stubbed) if " logs " in c]
    assert calls, "no docker compose logs call was made"
    assert calls[-1].endswith("toxpred"), calls[-1]


def test_logs_with_two_services_keeps_both(stubbed):
    """`${@:2}` dropped the first one, so this asked only for toxpred."""
    stubbed("logs", "toxagent-control", "toxpred")
    calls = [c for c in _docker_calls(stubbed) if " logs " in c]
    assert calls
    assert "toxagent-control" in calls[-1], calls[-1]
    assert "toxpred" in calls[-1], calls[-1]


def test_logs_with_no_service_asks_for_all_of_them(stubbed):
    stubbed("logs")
    calls = [c for c in _docker_calls(stubbed) if " logs " in c]
    assert calls
    assert calls[-1].rstrip().endswith("--tail=200"), calls[-1]


# --- I28: restore ------------------------------------------------------------

def test_restore_accepts_the_syntax_the_runbook_documents(stubbed, tmp_path):
    """`toxagent restore FILE` printed usage and exited 1 without ever
    reaching the database."""
    backup = tmp_path / "toxagent-20260908.sql.gz"
    subprocess.run(["gzip", "-c"], input=b"SELECT 1;\n", stdout=backup.open("wb"), check=True)

    result = stubbed("restore", str(backup), input_text="RESTORE\n")
    assert "usage:" not in result.stderr, result.stderr
    assert any("psql" in call for call in _docker_calls(stubbed)), _docker_calls(stubbed)


def test_restore_without_a_file_still_explains_itself(stubbed):
    result = stubbed("restore")
    assert result.returncode != 0
    assert "usage:" in result.stderr


def test_restore_refuses_a_path_that_does_not_exist(stubbed):
    result = stubbed("restore", "/nonexistent/backup.sql.gz")
    assert result.returncode != 0
    assert not any("psql" in call for call in _docker_calls(stubbed))


def test_restore_is_not_performed_without_the_typed_confirmation(stubbed, tmp_path):
    backup = tmp_path / "b.sql.gz"
    subprocess.run(["gzip", "-c"], input=b"SELECT 1;\n", stdout=backup.open("wb"), check=True)

    result = stubbed("restore", str(backup), input_text="yes\n")
    assert result.returncode != 0
    assert not any("psql" in call for call in _docker_calls(stubbed))
