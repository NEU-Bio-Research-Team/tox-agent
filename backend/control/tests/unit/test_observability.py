"""Logs a machine can read, with the credentials taken out (K12).

There was no logging configuration. Records went wherever uvicorn's default
handler put them, as prose, uncorrelated, and with nothing between a
credential and the log file — while `api/errors.py` was carefully keeping the
same class of value out of *responses*, on the grounds that "leaking an
unexpected exception's text is how a database URL or a provider key ends up in
a client's logs". It ends up in the server's logs the same way.
"""
from __future__ import annotations

import io
import json
import logging

import pytest

from toxagent import observability as obs


@pytest.fixture(autouse=True)
def clean_registry():
    obs.forget_secrets()
    yield
    obs.forget_secrets()
    logging.getLogger().handlers = []


def emit(record_call, *, secrets=(), level="INFO") -> list[dict]:
    stream = io.StringIO()
    obs.configure_logging(level=level, secrets=secrets, stream=stream)
    record_call(logging.getLogger("toxagent.test"))
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


# ------------------------------------------------------------------ shape


def test_a_line_is_one_json_object_with_the_fields_a_reader_needs():
    lines = emit(lambda log: log.info("run accepted"))
    assert len(lines) == 1
    line = lines[0]
    assert line["level"] == "INFO"
    assert line["logger"] == "toxagent.test"
    assert line["message"] == "run accepted"
    assert line["ts"]


def test_the_request_and_run_are_carried_without_every_call_site_passing_them():
    token = obs.request_id_var.set("req-1")
    run = obs.run_id_var.set("run_abc")
    try:
        line = emit(lambda log: log.info("tool started"))[0]
    finally:
        obs.request_id_var.reset(token)
        obs.run_id_var.reset(run)
    assert line["request_id"] == "req-1"
    assert line["run_id"] == "run_abc"
    assert "session_id" not in line, "an unset id is absent, not null"


def test_extra_fields_are_kept_and_standard_record_attributes_are_not():
    line = emit(lambda log: log.info("admitted", extra={"endpoint": "herg", "count": 3}))[0]
    assert line["endpoint"] == "herg"
    assert line["count"] == 3
    assert "msecs" not in line and "levelno" not in line


# -------------------------------------------------------------- redaction


@pytest.mark.parametrize(
    "text,must_not_contain",
    [
        ("Authorization: Bearer abc123def456ghi789", "abc123def456ghi789"),
        ("token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ4In0.sIgNaTuRe123", "sIgNaTuRe123"),
        ("using sk-proj-ABCdef1234567890", "sk-proj-ABCdef1234567890"),
        ("ghp-ABCdef1234567890 pushed", "ghp-ABCdef1234567890"),
        ("postgresql+asyncpg://toxagent:hunter2hunter2@db:5432/x", "hunter2hunter2"),
    ],
)
def test_a_credential_shape_never_reaches_the_stream(text, must_not_contain):
    line = emit(lambda log: log.info(text))[0]
    assert must_not_contain not in json.dumps(line)
    assert obs.REDACTED in line["message"]


def test_what_surrounds_a_credential_survives_it():
    """Redaction that eats the context makes the log useless and people turn
    it off. The host and the database name are not the secret."""
    line = emit(lambda log: log.info("connecting to postgresql://user:s3cretpass@db:5432/toxagent"))[0]
    assert "s3cretpass" not in line["message"]
    assert "db:5432/toxagent" in line["message"]
    assert "postgresql://user:" in line["message"]


def test_a_credential_in_the_arguments_is_scrubbed_too():
    """`log.info("calling %s", url)` puts it in record.args, not record.msg.
    A filter that only looked at the message would still write it out."""
    line = emit(lambda log: log.info("probing %s", "https://x.test?key=sk-live-ABCdef123456"))[0]
    assert "sk-live-ABCdef123456" not in json.dumps(line)


def test_a_credential_inside_a_traceback_is_scrubbed():
    def boom(log):
        try:
            raise RuntimeError("connect failed for postgresql://u:pa55word99@db/x")
        except RuntimeError:
            log.exception("startup failed")

    line = emit(boom)[0]
    assert "pa55word99" not in json.dumps(line)
    assert "exception" in line


def test_a_registered_secret_with_no_recognisable_shape_is_still_scrubbed():
    secret = "not-shaped-like-anything-in-particular-9f2a"
    line = emit(lambda log: log.info("signed with %s", secret), secrets=(secret,))[0]
    assert secret not in json.dumps(line)


def test_a_short_value_is_not_registered_as_a_secret():
    """Redacting a two-character string would redact half the log file."""
    obs.register_secret("ab")
    assert obs.redact("a sentence about ab and other things") == (
        "a sentence about ab and other things"
    )


def test_ordinary_text_is_left_alone():
    line = emit(lambda log: log.info("session ses_1 reached 3 runs, no runtime bound"))[0]
    assert line["message"] == "session ses_1 reached 3 runs, no runtime bound"


def test_a_library_that_logs_a_url_is_covered_too():
    """The filter is on the root handler, not on `toxagent.*`: uvicorn's
    access log and any dependency go through the same scrubbing."""
    stream = io.StringIO()
    obs.configure_logging(stream=stream)
    logging.getLogger("uvicorn.access").warning("retrying https://x.test with sk-live-ABCdef123456")
    assert "sk-live-ABCdef123456" not in stream.getvalue()
