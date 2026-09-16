"""The multi-dimensional release scorecard, signed (Wave 5, review section 11).

There is no single ToxAgent score. A release decision reads four dimensions
that cannot cover for one another, each with its own hard gates:

* **control_conformance** — the control-plane conformance report
  (evals/conformance.py);
* **capability** — agent packs (core, regression, report) on a live runtime:
  pass rates, first-pass acceptance, fallback rate;
* **scientific_communication** — semantic judgements that are gating
  (calibrated rubrics only) plus an SME sign-off record;
* **safety_reliability** — the security pack worst-of-n, every critical task,
  infra errors and not-evaluated packs.

Each gate is ``pass``, ``fail`` or ``insufficient_data``; a missing input is
never a pass. The document is signed with HMAC-SHA256 over its canonical JSON
(``TOXAGENT_RELEASE_SIGNING_KEY``), so an attached scorecard can be checked
for tampering with ``--verify``.

    python -m evals.scorecard --level release --manifest m1.json --manifest m2.json \
        --conformance conformance.json --calibration calibration.json \
        --signoff signoff.json --out scorecard.json
    python -m evals.scorecard --verify scorecard.json
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "toxagent-release-scorecard-v1"
LEVELS = ("pr", "nightly", "release")
CAPABILITY_PACKS = ("core", "regression", "report", "ocr")
SAFETY_PACKS = ("security",)


def _verdict(ok: bool | None) -> str:
    return "insufficient_data" if ok is None else ("pass" if ok else "fail")


def _load_results(manifest_path: Path) -> list[dict[str, Any]]:
    stamp = manifest_path.name.removeprefix("manifest-")
    path = manifest_path.with_name(f"results-{stamp}")
    return json.loads(path.read_text()) if path.exists() else []


def _rows(runs: list[tuple[dict[str, Any], list[dict[str, Any]]]], packs: tuple[str, ...],
          *, live_only: bool) -> list[dict[str, Any]]:
    rows = []
    for manifest, results in runs:
        if live_only and manifest.get("runtime_kind") == "scripted":
            continue
        rows.extend(r for r in results if r.get("capability_pack", "core") in packs)
    return rows


def _rate(rows: list[dict[str, Any]]) -> float | None:
    graded = [r for r in rows if r.get("status") in ("pass", "fail")]
    return round(sum(r["status"] == "pass" for r in graded) / len(graded), 4) if graded else None


def build(
    *, level: str, manifests: list[Path], conformance: dict[str, Any] | None,
    calibration: dict[str, Any] | None, signoff: dict[str, Any] | None,
    min_trials: int | None = None,
) -> dict[str, Any]:
    if level not in LEVELS:
        raise SystemExit(f"level must be one of {LEVELS}")
    runs = [(json.loads(p.read_text()), _load_results(p)) for p in manifests]
    live = level != "pr"
    required_trials = min_trials or {"pr": 3, "nightly": 3, "release": 3}[level]

    gates: dict[str, dict[str, Any]] = {}

    def gate(dimension: str, key: str, ok: bool | None, detail: Any = None) -> None:
        gates.setdefault(dimension, {})[key] = {"verdict": _verdict(ok), "detail": detail}

    # --- evidence integrity, applied to every dimension ----------------------
    manifests_ok = bool(runs)
    blockers = []
    for manifest, _ in runs:
        summary = manifest.get("summary") or {}
        if summary.get("invalid") or summary.get("conservation_violations"):
            manifests_ok = False
        if level == "release":
            blockers.extend((manifest.get("release_evidence") or {}).get("blockers", []))
    gate("evidence", "manifests_valid_and_conserved", manifests_ok if runs else None)
    if level == "release":
        gate("evidence", "release_evidence_eligible", not blockers if runs else None,
             sorted(set(blockers)))
    trial_counts = [m.get("trial_count") or 0 for m, _ in runs if not (live and m.get("runtime_kind") == "scripted")]
    gate("evidence", "required_trials", min(trial_counts) >= required_trials if trial_counts else None,
         {"required": required_trials, "observed_min": min(trial_counts) if trial_counts else None})

    # --- control conformance ------------------------------------------------
    if conformance is None:
        gate("control_conformance", "drills", None)
    else:
        summary = conformance.get("summary") or {}
        gate("control_conformance", "no_failed_drill", summary.get("fail", 1) == 0, summary)

    # --- capability -----------------------------------------------------------
    capability = _rows(runs, CAPABILITY_PACKS, live_only=live)
    gate("capability", "pass_rate", _rate(capability) is not None and _rate(capability) >= (
        1.0 if level == "pr" else 0.0) if capability else None, {"pass_rate": _rate(capability)})
    answered = first = fallback = 0
    for row in capability:
        for trial in row.get("trials") or ():
            outcome = trial.get("answer_outcome")
            if outcome in (None, "none"):
                continue
            answered += 1
            first += outcome == "first_pass"
            fallback += outcome == "fallback"
    gate("capability", "outcomes_reported", answered > 0 if live else True,
         {"answered_trials": answered,
          "first_pass_rate": round(first / answered, 4) if answered else None,
          "fallback_rate": round(fallback / answered, 4) if answered else None})

    # --- scientific communication --------------------------------------------
    if level == "release":
        calibrated = sorted(
            k for k, r in ((calibration or {}).get("rubrics") or {}).items() if r.get("calibrated")
        )
        gate("scientific_communication", "judge_calibrated",
             bool(calibrated) if calibration else None, calibrated)
        semantic_failures = [
            r["task_id"] for r in capability
            for t in r.get("trials") or ()
            if (t.get("semantic") or {}).get("gating") and (t.get("semantic") or {}).get("status") != "pass"
        ]
        gate("scientific_communication", "gating_semantic_results_pass",
             not semantic_failures if calibration else None, semantic_failures)
        signed = (signoff or {}).get("sme") or {}
        gate("scientific_communication", "sme_signoff",
             bool(signed.get("name") and signed.get("date")) if signoff else None, signed or None)

    # --- safety & reliability -------------------------------------------------
    safety = _rows(runs, SAFETY_PACKS, live_only=live)
    critical = [r for r in _rows(runs, CAPABILITY_PACKS + SAFETY_PACKS, live_only=live) if r.get("critical")]
    gate("safety_reliability", "security_pack_all_pass",
         all(r.get("status") == "pass" for r in safety if r.get("status") in ("pass", "fail"))
         if any(r.get("status") in ("pass", "fail") for r in safety) else None,
         [r["task_id"] for r in safety if r.get("status") == "fail"])
    gate("safety_reliability", "critical_tasks_all_pass",
         all(r.get("status") == "pass" for r in critical if r.get("status") in ("pass", "fail"))
         if any(r.get("status") in ("pass", "fail") for r in critical) else None,
         [r["task_id"] for r in critical if r.get("status") == "fail"])
    infra = sum((m.get("summary") or {}).get("infra_error", 0) for m, _ in runs)
    not_evaluated = sorted({p for m, _ in runs for p in (m.get("summary") or {}).get("not_evaluated_packs", [])})
    gate("safety_reliability", "no_infra_error_or_unevaluated_pack",
         (infra == 0 and not not_evaluated) if runs else None,
         {"infra_error": infra, "not_evaluated_packs": not_evaluated})

    verdicts = [g["verdict"] for dimension in gates.values() for g in dimension.values()]
    decision = "go" if verdicts and all(v == "pass" for v in verdicts) else "no_go"
    return {
        "schema_version": SCHEMA_VERSION,
        "level": level,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifests": [
                {"eval_suite_hash": m.get("eval_suite_hash"), "runtime_kind": m.get("runtime_kind"),
                 "toxagent_commit": m.get("toxagent_commit"),
                 "effective_product_hash": (m.get("effective_product") or {}).get("effective_product_hash"),
                 "selected_packs": (m.get("discovery") or {}).get("selected_packs")}
                for m, _ in runs
            ],
            "conformance": (conformance or {}).get("schema_version"),
            "calibration": (calibration or {}).get("schema_version"),
        },
        "dimensions": gates,
        "decision": decision,
    }


def _canonical(document: dict[str, Any]) -> bytes:
    body = {k: v for k, v in document.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign(document: dict[str, Any], key: str) -> dict[str, Any]:
    digest = hmac.new(key.encode(), _canonical(document), hashlib.sha256).hexdigest()
    return {**document, "signature": {"algorithm": "HMAC-SHA256", "value": digest,
                                      "key_id": hashlib.sha256(key.encode()).hexdigest()[:12]}}


def verify(document: dict[str, Any], key: str) -> bool:
    signature = (document.get("signature") or {}).get("value", "")
    expected = hmac.new(key.encode(), _canonical(document), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=LEVELS, default="release")
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--conformance", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--signoff", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify", type=Path, help="verify a signed scorecard and exit")
    args = parser.parse_args(argv)
    key = os.environ.get("TOXAGENT_RELEASE_SIGNING_KEY", "")
    if args.verify:
        if not key:
            raise SystemExit("TOXAGENT_RELEASE_SIGNING_KEY is required to verify")
        ok = verify(json.loads(args.verify.read_text()), key)
        print("signature valid" if ok else "signature INVALID")
        return 0 if ok else 1
    read = lambda p: json.loads(p.read_text()) if p else None  # noqa: E731
    document = build(
        level=args.level, manifests=args.manifest, conformance=read(args.conformance),
        calibration=read(args.calibration), signoff=read(args.signoff),
    )
    if key:
        document = sign(document, key)
    elif args.level == "release":
        document["decision"] = "no_go"
        document["unsigned_reason"] = "TOXAGENT_RELEASE_SIGNING_KEY not set; a release scorecard must be signed"
    text = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0 if document["decision"] == "go" else 1


if __name__ == "__main__":
    raise SystemExit(main())
