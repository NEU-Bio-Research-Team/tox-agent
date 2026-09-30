"""Describe a study run: what happened, never how good it was.

    python -m evals.investigation.report --study pilot-2026-09-25

Writes ``<study>/study-report.md`` and ``study-report.json``: per system the
denominators (ok / error / pending out of expected), time per turn, tokens and
cost where the platform reported them, and for ToxAgent arms the product's own
process facts — tool calls, fallback answers, stop reasons, case size — and,
for the skill arms, how skills were triggered against each skill's declared
positive and negative case tags (RETHINK §5.5: trigger precision and recall,
false triggers, misses).

None of this is a quality score. Quality is what the lab grades; this report
exists so the grades can be read next to cost, latency and behaviour.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evals.investigation import cases as case_module
from evals.investigation.record import StudyStore, answering_models
from evals.investigation.run import DEFAULT_ROOT

SCHEMA_VERSION = "investigation-study-report-v1"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


#: A platform declining to answer is something a researcher using it would
#: see, so it is reported apart from quota and transport failures, over every
#: attempt: a refusal that a rerun got past still happened.
_ERROR_CLASSES = (
    ("provider_refusal", ("safeguards flagged", "legal/aup", "usage policy")),
    ("quota_or_capacity", ("session limit", "rate limit", "429", "503", "quota", "overload")),
)


def error_class(error: str | None) -> str:
    text = (error or "").lower()
    for name, needles in _ERROR_CLASSES:
        if any(needle in text for needle in needles):
            return name
    return "other"


def skill_triggers(records, cases: dict[str, dict[str, Any]], skills: list[Any]) -> dict[str, Any]:
    """Per skill: on how many cases it was read, against the cases its manifest
    says should (positive tags) and should not (negative tags) trigger it."""
    out: dict[str, Any] = {}
    for skill in skills:
        positive = set(skill.manifest["eval_set"]["positive_tags"])
        negative = set(skill.manifest["eval_set"]["negative_tags"])
        loaded_on, should, should_not = set(), set(), set()
        for record in records:
            tags = set(cases[record.case_id]["tags"])
            if tags & positive:
                should.add(record.case_id)
            elif tags & negative:
                should_not.add(record.case_id)
            for turn in record.turns:
                loaded = (turn.meta.get("skills") or {}).get("loaded") or []
                if any(item.get("skill_id") == skill.skill_id for item in loaded):
                    loaded_on.add(record.case_id)
        hits = loaded_on & should
        out[skill.skill_id] = {
            "cases_read": sorted(loaded_on),
            "positive_cases": sorted(should), "negative_cases": sorted(should_not),
            "trigger_precision": round(len(hits) / len(loaded_on), 3) if loaded_on else None,
            "trigger_recall": round(len(hits) / len(should), 3) if should else None,
            "false_triggers_on_negative_cases": sorted(loaded_on & should_not),
            "misses": sorted(should - loaded_on),
        }
    return out


def build_report(study_dir: Path, *, cases_dir: Path = case_module.CASES_DIR) -> dict[str, Any]:
    from toxagent.application.investigation.skill_catalog import load_catalog
    from toxagent.platform.config import PACKAGE_ROOT

    store = StudyStore(study_dir)
    manifest = store.manifest()
    latest = store.latest_records()
    all_records = store.records()
    cases = {c["case_id"]: c for c in case_module.load_cases(cases_dir)}
    skills = list(load_catalog(PACKAGE_ROOT / "agent_profiles").skills)
    expected_per_system = len(manifest.get("case_set", {}).get("cases", {})) * int(manifest.get("trials", 1))
    by_system: dict[str, list] = defaultdict(list)
    for record in latest.values():
        by_system[record.system_id].append(record)
    attempts = Counter(r.system_id for r in all_records)

    systems: dict[str, Any] = {}
    for system_id in sorted(manifest.get("systems", {})):
        records = by_system.get(system_id, [])
        statuses = Counter(r.status for r in records)
        ok = [r for r in records if r.status == "ok"]
        durations = [t.duration_s for r in ok for t in r.turns if isinstance(t.duration_s, (int, float))]
        usage_keys = sorted({k for r in ok for k, v in (r.usage or {}).items() if isinstance(v, (int, float))})
        entry: dict[str, Any] = {
            "arm": manifest["systems"][system_id].get("rethink_arm"),
            "expected": expected_per_system,
            "ok": statuses.get("ok", 0), "error": statuses.get("error", 0),
            "pending": statuses.get("pending", 0),
            "not_run": expected_per_system - len(records),
            "attempts_recorded": attempts.get(system_id, 0),
            "models": sorted({m for r in ok for m in answering_models(
                r, store, (manifest["systems"][system_id].get("adapter_config") or {}).get("model_requested"))}),
            "median_turn_seconds": _median(durations),
            "usage_totals": {k: round(sum((r.usage or {}).get(k, 0) for r in ok), 6) for k in usage_keys},
            "errors": {r.case_id: r.error for r in records if r.status == "error"},
            "error_attempts_by_class": dict(Counter(
                error_class(r.error) for r in all_records
                if r.system_id == system_id and r.status == "error")),
        }
        if manifest["systems"][system_id].get("family") == "toxagent":
            turns = [t for r in ok for t in r.turns]
            entry["toxagent"] = {
                "turns": len(turns),
                "fallback_turns": sum(1 for t in turns if t.meta.get("is_fallback")),
                "median_tool_calls_per_turn": _median([len(t.meta.get("tool_calls") or []) for t in turns]),
                "stop_reasons": dict(Counter(str(t.meta.get("stop_reason")) for t in turns)),
                "tool_use": dict(Counter(name for t in turns for name in t.meta.get("tool_calls") or [])),
            }
            if manifest["systems"][system_id].get("skills_mode") == "dynamic":
                entry["skill_triggers"] = skill_triggers(ok, cases, skills)
        systems[system_id] = entry
    return {
        "schema_version": SCHEMA_VERSION, "study_id": manifest.get("study_id"),
        "case_set_sha256": manifest.get("case_set", {}).get("sha256"),
        "environment": manifest.get("environment"), "systems": systems,
        "note": "Process description only; quality is graded by the lab.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Study report — {report['study_id']}", "",
             report["note"], "",
             "| System | Arm | ok / error / pending / not run (of expected) | Failed attempts by class (all attempts) "
             "| Model(s) | Median s per turn | Usage (reported) |",
             "|---|---|---|---|---|---|---|"]
    for system_id, s in report["systems"].items():
        usage = ", ".join(f"{k}={v:g}" for k, v in s["usage_totals"].items()) or "—"
        failed = ", ".join(f"{k}: {v}" for k, v in sorted(s["error_attempts_by_class"].items())) or "—"
        lines.append(
            f"| `{system_id}` | {s['arm']} | {s['ok']} / {s['error']} / {s['pending']} / {s['not_run']} "
            f"({s['expected']}) | {failed} | {', '.join(s['models']) or '—'} | {s['median_turn_seconds'] or '—'} "
            f"| {usage} |"
        )
    toxagent = {k: v for k, v in report["systems"].items() if "toxagent" in v}
    if toxagent:
        lines += ["", "## ToxAgent arms: process", "",
                  "| System | Turns | Fallback answers | Median tool calls / turn | Stop reasons |",
                  "|---|---|---|---|---|"]
        for system_id, s in toxagent.items():
            t = s["toxagent"]
            reasons = ", ".join(f"{k}: {v}" for k, v in sorted(t["stop_reasons"].items()))
            lines.append(f"| `{system_id}` | {t['turns']} | {t['fallback_turns']} | "
                         f"{t['median_tool_calls_per_turn']} | {reasons} |")
    for system_id, s in report["systems"].items():
        if "skill_triggers" not in s:
            continue
        lines += ["", f"## Skill triggering — `{system_id}`", "",
                  "| Skill | Cases read | Trigger precision | Trigger recall | False triggers (negative cases) | Misses |",
                  "|---|---|---|---|---|---|"]
        for skill_id, k in s["skill_triggers"].items():
            lines.append(f"| `{skill_id}` | {len(k['cases_read'])} | {k['trigger_precision']} | "
                         f"{k['trigger_recall']} | {', '.join(k['false_triggers_on_negative_cases']) or '—'} | "
                         f"{', '.join(k['misses']) or '—'} |")
    errors = {k: v["errors"] for k, v in report["systems"].items() if v["errors"]}
    if errors:
        lines += ["", "## Errors (latest attempt)", ""]
        for system_id, items in errors.items():
            for case_id, error in sorted(items.items()):
                lines.append(f"- `{system_id}` / {case_id}: {error}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cases-dir", type=Path, default=case_module.CASES_DIR)
    args = parser.parse_args(argv)
    study_dir = args.root / args.study
    report = build_report(study_dir, cases_dir=args.cases_dir)
    (study_dir / "study-report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    text = render_markdown(report)
    (study_dir / "study-report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
