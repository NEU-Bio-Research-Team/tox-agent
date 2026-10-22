"""Re-read what the researcher was shown, for runs recorded with adapter v1.

``toxagent-replay-adapter-v1`` matched a run's assistant messages on a
``run_id`` field the API's ``Message`` does not have, so every run whose answer
was not published was recorded with an empty reply (or only a fallback's
text), although the product had shown the reader the server's confirmed facts
and why nothing was released. Published answers were read from the answer and
are unaffected.

This does not run any arm again. For each recorded turn whose disposition was
not ``publish``, it reads the session's transcript back from the deployment
that served it -- the product's own persisted record of what it showed -- and
renders it with the fixed matcher. The source study is never modified: the
result is a derived study beside it, whose records differ only in those
turns' ``response_text`` (each amended turn says so in its ``meta``), with a
``recapture.json`` listing every amendment. A turn whose transcript cannot be
read is left as it was and listed as such.

    python -m evals.investigation.recapture --study <id> --derived <id> \\
        --arm B2_fixed_workflow=http://127.0.0.1:8021 ... --token <token>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx

from evals.investigation.adapters.toxagent_replay import (
    ADAPTER_VERSION,
    messages_for_run,
    visible_text,
)
from evals.investigation.record import now_iso
from evals.investigation.run import DEFAULT_ROOT

SCHEMA_VERSION = "recapture-v1"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def recapture(*, source: Path, derived: Path, urls: dict[str, str], token: str,
              client: httpx.Client | None = None) -> dict[str, Any]:
    if derived.exists():
        raise SystemExit(f"{derived} exists; a derived study is never overwritten")
    http = client or httpx.Client(timeout=30.0)
    headers = {"Authorization": f"Bearer {token}"}
    amendments: list[dict[str, Any]] = []
    unread: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    transcripts: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for line in (source / "records.jsonl").read_text().splitlines():
        record = json.loads(line)
        records.append(record)
        url = urls.get(record["system_id"])
        trace_rel = (record.get("artifacts") or {}).get("trace")
        if not url or not trace_rel:
            continue
        trace = json.loads((source / trace_rel).read_text())
        session_id = trace.get("session_id")
        runs = {int(item.get("turn", i)): item for i, item in enumerate(trace.get("runs") or ())}
        for turn in record.get("turns") or ():
            meta = turn.get("meta") or {}
            if not meta.get("disposition") or meta.get("disposition") == "publish":
                continue
            run = runs.get(int(turn["index"])) or {}
            run_id = meta.get("run_id")
            where = {"record_id": record.get("record_id"), "system_id": record["system_id"],
                     "case_id": record["case_id"], "turn": turn["index"], "run_id": run_id}
            key = (record["system_id"], session_id)
            if key not in transcripts:
                response = http.get(f"{url}/v1/sessions/{session_id}/messages",
                                    params={"limit": 500}, headers=headers)
                if response.status_code != 200:
                    unread.append({**where, "status": response.status_code})
                    continue
                transcripts[key] = response.json().get("messages") or []
            answer = run.get("answer")
            found = messages_for_run(transcripts[key], run_id, (answer or {}).get("answer_id"))
            text = visible_text(answer, found, trace.get("evidence") or [], published=False)
            before = turn.get("response_text") or ""
            if not found:
                unread.append({**where, "status": "no message names this run"})
                continue
            if text == before:
                continue
            turn["response_text"] = text
            turn["meta"] = {**meta, "recaptured": {
                "schema_version": SCHEMA_VERSION, "adapter": ADAPTER_VERSION,
                "from": f"GET /v1/sessions/{session_id}/messages",
                "previous_text_sha256": _sha(before), "text_sha256": _sha(text),
            }}
            amendments.append({**where, "previous_text": before, "text": text,
                               "message_ids": [m.get("message_id") for m in found]})
        if record.get("turns"):
            record["final_text"] = record["turns"][-1].get("response_text") or ""

    derived.mkdir(parents=True)
    for name in ("raw", "products"):
        if (source / name).exists():
            shutil.copytree(source / name, derived / name)
    manifest = json.loads((source / "study-manifest.json").read_text())
    manifest["study_id"] = derived.name
    manifest["derived_from"] = {
        "study_id": source.name, "schema_version": SCHEMA_VERSION, "at": now_iso(),
        "why": "adapter v1 recorded no reader text for runs that did not publish",
        "amended_turns": len(amendments), "unread_turns": len(unread),
        "agents_rerun": 0,
    }
    (derived / "study-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (derived / "records.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    report = {"schema_version": SCHEMA_VERSION, "source": source.name, "derived": derived.name,
              "at": now_iso(), "adapter": ADAPTER_VERSION, "amendments": amendments,
              "unread": unread}
    (derived / "recapture.json").write_text(json.dumps(report, indent=2) + "\n")
    return {"derived": str(derived), "amended": len(amendments), "unread": len(unread)}


def _pairs(values: list[str]) -> dict[str, str]:
    return dict(value.split("=", 1) for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--study", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--arm", action="append", default=[], metavar="SYSTEM=URL")
    parser.add_argument("--token", default="")
    args = parser.parse_args(argv)
    out = recapture(source=args.root / args.study, derived=args.root / args.derived,
                    urls=_pairs(args.arm), token=args.token)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
