"""Logs a machine can read and a person can safely keep (K12).

The control plane had no logging configuration at all. Records went wherever
uvicorn's default handler put them, as prose, with no correlation between the
lines belonging to one request and nothing standing between a credential and
the log file. The error envelope already refuses to put an unexpected
exception's text in a *response* — "leaking an unexpected exception's text is
how a database URL or a provider key ends up in a client's logs" — and the
same reasoning applies to the server's own logs, where nothing was doing it.

Two things here:

*Structure.* One JSON object per line, with the request id, and the session
and run when the handler knows them. Correlating a failure with the request
that caused it stops being a matter of reading timestamps.

*Redaction.* A filter over every record, its arguments and its exception text,
applied where records are emitted rather than at each of the call sites — a
redaction policy that depends on 23 call sites remembering it is a policy that
holds until the twenty-fourth.

What redaction here is and is not: it catches credential *shapes* — bearer
tokens, JWTs, `sk-`-prefixed keys, connection-string passwords, and any value
registered as a secret at startup. It is a second line of defence. The first
is not putting a credential in a log call, and this does not excuse that.
"""
from __future__ import annotations

import contextvars
import json
import logging
import re
import uuid
from typing import Any, Iterable

#: Set per request, so a log line names the request that produced it without
#: every call site passing an id down. A ContextVar rather than a thread-local
#: because the server is async: one thread serves many requests.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "toxagent_request_id", default=None
)
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "toxagent_session_id", default=None
)
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "toxagent_run_id", default=None
)

REDACTED = "[redacted]"

#: Credential shapes, not credential names. A rule keyed on the *word* next to
#: a secret misses `{"value": "sk-..."}` and every other spelling; a rule keyed
#: on the shape of the secret itself does not.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # `Authorization: Bearer <token>` in any casing, and the same shape inside
    # a serialized header dict.
    ("bearer", re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}")),
    # A JWT: three base64url segments. Capability tokens and user tokens both
    # look like this, and both are credentials.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9._-]{8,}\.[A-Za-z0-9._-]{4,}")),
    # Provider API keys. `sk-` covers OpenAI and the OpenAI-compatible
    # providers this control plane speaks to; the others are common enough to
    # be worth catching when someone points it at a gateway.
    ("api_key", re.compile(r"\b(?:sk|rk|pk|xoxb|ghp|gho|glpat)-[A-Za-z0-9_-]{8,}")),
    # The password inside a database or broker URL.
    ("url_password", re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/@]+):[^\s@]+@")),
)

#: Values registered at startup — the capability secret, the database
#: password. Matched literally, because they have no shape to recognise.
_literals: set[str] = set()


def register_secret(value: str | None) -> None:
    """Add a known secret so it is scrubbed wherever it appears.

    Short values are ignored: a two-character "secret" would redact half the
    log file, which loses more than it protects.
    """
    if value and len(value) >= 8:
        _literals.add(value)


def forget_secrets() -> None:
    """Only for tests; a running process never unlearns a secret."""
    _literals.clear()


def redact(text: str) -> str:
    for value in _literals:
        text = text.replace(value, REDACTED)
    for name, pattern in PATTERNS:
        if name == "bearer":
            text = pattern.sub(rf"\1 {REDACTED}", text)
        elif name == "url_password":
            text = pattern.sub(rf"\1:{REDACTED}@", text)
        else:
            text = pattern.sub(REDACTED, text)
    return text


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_redact_value(v) for v in value)
    return value


class RedactionFilter(logging.Filter):
    """Scrub the message, its arguments and any exception text.

    The arguments matter as much as the message: `log.info("calling %s", url)`
    puts the credential in `record.args`, not in `record.msg`, and a filter
    that only looked at the formatted message would still write it out
    whenever a handler formatted the record itself.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = _redact_value(record.args)
        if record.exc_info and record.exc_info[1] is not None:
            record.exc_text = redact(
                record.exc_text or logging.Formatter().formatException(record.exc_info)
            )
            record.exc_info = None
        return True


#: Attributes `logging` puts on every record. Anything else a caller attached
#: with `extra=` is context worth keeping, and is emitted alongside.
_STANDARD = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for name, var in (
            ("request_id", request_id_var),
            ("session_id", session_id_var),
            ("run_id", run_id_var),
        ):
            value = var.get()
            if value:
                payload[name] = value
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = _redact_value(value)
        if record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    *, level: str = "INFO", secrets: Iterable[str | None] = (), stream: Any = None
) -> None:
    """Install the JSON formatter and the redaction filter on the root logger.

    On the root, and on the handlers uvicorn has already installed: attaching
    the filter to `toxagent.*` alone would leave `uvicorn.access` and any
    library that logs a URL writing whatever they like.
    """
    for secret in secrets:
        register_secret(secret)

    handler = logging.StreamHandler(stream) if stream is not None else logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactionFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.addFilter(RedactionFilter())


def new_request_id() -> str:
    return uuid.uuid4().hex
