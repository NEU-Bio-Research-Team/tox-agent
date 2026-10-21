"""What one study writes: an append-only record per (case, system, trial), raw
artifacts beside it, and a study manifest.

Everything a lab grader, an auditor or a later re-analysis could need is kept:
the exact text sent (with the system prompt and any stitched transcript), the
raw output, the model id the platform *reported* (not the one we asked for),
timings, usage and cost where the platform reports them, and for ToxAgent the
full product trace (runs, tool calls, answers, decision states, case, dossiers).

Records are appended, never rewritten; a later record for the same key
supersedes an earlier one, and ``latest_records`` applies that rule, so a
failed trial that was re-run leaves both attempts on disk.

Protocol v2 (#50, master plan §13.6) changes the rule for what is scored: a
(case, system, trial) has one **primary** attempt -- the first one recorded --
and that is what the packet and the scorer read (``primary_records``). A
rerun is a ``diagnostic_rerun``: kept, costed, reported, and never a
replacement for the primary outcome, so a failure cannot be rerun into a
success. Each attempt writes its raw artifacts under its own attempt id, and
a raw file is never overwritten.
"""
from __future__ import annotations

import fcntl
import json
import os
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

RECORD_SCHEMA = "investigation-run-v1"
MANIFEST_SCHEMA = "investigation-study-manifest-v1"

#: Protocol v2: which attempt of a (case, system, trial) a record is.
ATTEMPT_PRIMARY = "primary"
ATTEMPT_DIAGNOSTIC = "diagnostic_rerun"

STATUS_OK = "ok"
STATUS_ERROR = "error"
#: A manual adapter is waiting for a response file: not a failure, not a result.
STATUS_PENDING = "pending"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    index: int
    #: The researcher's message for this turn, as the case states it.
    user_text: str
    #: Exactly what the system received: for a platform the full prompt with
    #: the preamble, snapshot and stitched transcript; for ToxAgent the message.
    sent_text: str
    response_text: str = ""
    started_at: str = ""
    ended_at: str = ""
    duration_s: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    study_id: str
    case_id: str
    case_sha256: str
    system_id: str
    arm: dict[str, Any]
    trial: int
    status: str
    started_at: str
    ended_at: str
    model: dict[str, Any] = field(default_factory=dict)
    turns: list[TurnRecord] = field(default_factory=list)
    final_text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    toxagent: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: str = RECORD_SCHEMA
    #: Protocol v2: the attempt's id (its raw directory), and whether it is
    #: the primary attempt or a diagnostic rerun. ``None`` on a v1 record.
    attempt_id: str | None = None
    attempt_kind: str | None = None
    #: Protocol v2: the sha256 of the inputs file every arm was given.
    inputs_sha256: str | None = None

    def key(self) -> tuple[str, str, int]:
        return (self.case_id, self.system_id, self.trial)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        turns = [TurnRecord(**t) for t in data.get("turns") or ()]
        return cls(**{**data, "turns": turns})


class StudyStore:
    """One directory per study: records.jsonl, raw/, study-manifest.json."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.records_path = self.root / "records.jsonl"
        self.manifest_path = self.root / "study-manifest.json"

    def raw_dir(self, system_id: str, case_id: str, trial: int,
                attempt_id: str | None = None) -> Path:
        path = self.root / "raw" / system_id / case_id / f"t{trial}"
        if attempt_id is not None:
            path = path / attempt_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_raw(self, system_id: str, case_id: str, trial: int, name: str, content: Any,
                  *, attempt_id: str | None = None) -> str:
        path = self.raw_dir(system_id, case_id, trial, attempt_id) / name
        if attempt_id is not None and path.exists():
            # An attempt's raw record is evidence; a second write to it would
            # replace what the first attempt saw.
            raise FileExistsError(f"{path} exists; raw attempt files are never overwritten")
        if isinstance(content, (bytes, bytearray)):
            path.write_bytes(content)
        elif isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
        return str(path.relative_to(self.root))

    def append(self, record: RunRecord) -> None:
        """One record, one locked ``write`` on an ``O_APPEND`` descriptor, so
        several runner processes can share a study without interleaving lines."""
        self.root.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(record.to_dict(), ensure_ascii=False, default=str) + "\n").encode("utf-8")
        fd = os.open(self.records_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def records(self) -> list[RunRecord]:
        if not self.records_path.exists():
            return []
        return [
            RunRecord.from_dict(json.loads(line))
            for line in self.records_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    def latest_records(self) -> dict[tuple[str, str, int], RunRecord]:
        latest: dict[tuple[str, str, int], RunRecord] = {}
        for record in self.records():
            latest[record.key()] = record
        return latest

    def primary_records(self) -> dict[tuple[str, str, int], RunRecord]:
        """Protocol v2: the first non-diagnostic attempt per key -- the one
        scored, whatever a later rerun did."""
        primary: dict[tuple[str, str, int], RunRecord] = {}
        for record in self.records():
            if record.attempt_kind == ATTEMPT_DIAGNOSTIC:
                continue
            primary.setdefault(record.key(), record)
        return primary

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(f".tmp{os.getpid()}")
        temporary.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def update_manifest(self, change) -> dict[str, Any]:
        """Read, change and write the manifest under a lock, so a second runner
        on the same study merges into it instead of overwriting it."""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".manifest.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                manifest = change(self.manifest())
                self.write_manifest(manifest)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return manifest

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def environment() -> dict[str, Any]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Untracked files are the workspace's business; a modified tracked
        # file is a code state no commit describes, so it is recorded.
        "git_dirty_tracked": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
    }


def summarise(records: Iterable[RunRecord]) -> dict[str, Any]:
    """Denominators per system: never a pass rate, only what exists."""
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        row = summary.setdefault(record.system_id, {STATUS_OK: 0, STATUS_ERROR: 0, STATUS_PENDING: 0})
        row[record.status] = row.get(record.status, 0) + 1
    return summary


def answering_models(record: RunRecord, store: StudyStore, requested: str | None = None) -> list[str]:
    """The model(s) that wrote this record's answers, re-derived from the raw
    per-model usage where the adapter stored it (the claude CLI), else the
    model the record carries. Lets an improved resolver correct the reading of
    a run without re-running it."""
    from evals.investigation.adapters.platform_cli import _main_model

    raw_path = store.root / record.artifacts["raw"] if "raw" in record.artifacts else None
    if raw_path is not None and raw_path.exists():
        found = []
        for turn in json.loads(raw_path.read_text()):
            usage = ((turn.get("result") or {}).get("modelUsage")) if isinstance(turn, dict) else None
            if usage:
                found.append(str(_main_model(usage, requested)))
        if found:
            return sorted(set(found))
    resolved = (record.model or {}).get("model_id_resolved")
    if isinstance(resolved, list):
        return sorted(str(m) for m in resolved)
    return [str(resolved)] if resolved else []
