"""The capability matrix: what a default deployment really offers, from code.

RETHINK §6 P0 asks for "one source describing the deployment and the skill
surface" before any capability is claimed. ``architecture_inventory`` states
the product's *shape* (intents, queues, packs); this states its *capability*:
which predictor models are served or blocked and why, what the explainer has
been measured to do, which tools each profile really exposes by default and
which flag would add more, and what every instruction surface costs in prompt
tokens.

Everything is derived from the objects the product composes itself from — the
predictor registry manifests, ``toxagent.flags``, ``tools.registry``, the agent
profile files, ``domain.explainer_validation`` — so the checked-in output
cannot quietly disagree with the code. ``tests/unit/test_capability_matrix.py``
fails on drift. No timestamp: the output is deterministic.

    python -m evals.capability_matrix           # print the JSON
    python -m evals.capability_matrix --write   # regenerate docs/capability-matrix.json + CAPABILITY_MATRIX.md
    python -m evals.capability_matrix --check   # exit 1 on drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
JSON_PATH = REPO_ROOT / "docs" / "capability-matrix.json"
MARKDOWN_PATH = REPO_ROOT / "docs" / "CAPABILITY_MATRIX.md"
MODEL_MANIFESTS = REPO_ROOT / "backend" / "predictor" / "registry" / "models"
SCHEMA_VERSION = "capability-matrix-v1"


def _predictor_models() -> list[dict[str, Any]]:
    import yaml

    models: list[dict[str, Any]] = []
    for path in sorted(MODEL_MANIFESTS.glob("*.yaml")):
        for entry in yaml.safe_load(path.read_text()).get("models", ()):
            blocked = " ".join(str(entry.get("blocked_reason") or "").split())
            base = entry.get("base_model") or {}
            models.append({
                "model_id": entry["model_id"],
                "display_name": entry.get("display_name"),
                "endpoints": list(entry.get("capabilities") or ()),
                "status": "blocked" if blocked else "served",
                "blocked_reason": blocked or None,
                "required": bool(entry.get("required")),
                "feature_schema_version": entry.get("feature_schema_version"),
                "base_model": base.get("id"),
                "manifest": str(path.relative_to(REPO_ROOT)),
            })
    return models


def _endpoints(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    endpoints: dict[str, dict[str, Any]] = {}
    for model in models:
        for endpoint in model["endpoints"]:
            current = endpoints.get(endpoint)
            # A served model wins over a blocked one for the same endpoint.
            if current is None or (current["status"] == "blocked" and model["status"] == "served"):
                endpoints[endpoint] = {"status": model["status"], "model_id": model["model_id"]}
    return dict(sorted(endpoints.items()))


def _tool_profiles() -> dict[str, dict[str, Any]]:
    from toxagent.tools.registry import FLAG_GATED_TOOLS, PROFILES

    return {
        name: {
            "default": sorted(t for t in tools if t not in FLAG_GATED_TOOLS),
            "flag_gated": {t: FLAG_GATED_TOOLS[t] for t in sorted(tools) if t in FLAG_GATED_TOOLS},
        }
        for name, tools in sorted(PROFILES.items())
    }


def _flags() -> dict[str, dict[str, Any]]:
    from toxagent import flags as rollout
    from toxagent.tools.registry import FLAG_GATED_TOOLS

    return {
        f.name: {
            "default": f.default,
            "owner": f.owner,
            "remove_by": f.remove_by.isoformat(),
            "gates_tools": sorted(t for t, name in FLAG_GATED_TOOLS.items() if name == f.name),
            "description": " ".join(f.description.split()),
        }
        for f in sorted(rollout.FLAGS, key=lambda f: f.name)
    }


def _decision_support_prompt() -> dict[str, Any]:
    from toxagent.harness import context
    from toxagent.harness.prompt_budget import estimate_tokens

    components = {
        "product_role": context.PRODUCT_ROLE,
        "scientific_invariants": context.SCIENTIFIC_INVARIANTS,
        "answer_format": context.ANSWER_FORMAT,
        "required_limitations_guide": context.REQUIRED_LIMITATIONS_GUIDE,
        "decision_support_policy": context.DECISION_SUPPORT_POLICY,
    }
    return {
        "loading": "static: every component is composed into every decision_support turn",
        "components": {
            name: {"characters": len(text), "estimated_tokens": estimate_tokens(text)}
            for name, text in components.items()
        },
        "estimated_tokens_total": sum(estimate_tokens(t) for t in components.values()),
    }


def _report_build_skills() -> dict[str, Any]:
    from toxagent.config import PACKAGE_ROOT
    from toxagent.harness.prompt_budget import estimate_tokens
    from toxagent.harness.report_profile import compose_report_profile

    profiles_dir = PACKAGE_ROOT / "agent_profiles"
    composed = compose_report_profile(profiles_dir)
    root = profiles_dir / "report_build"
    files: dict[str, dict[str, Any]] = {}
    for relative in sorted(composed.file_hashes):
        text = (root / relative).read_text(encoding="utf-8")
        files[relative] = {"characters": len(text), "estimated_tokens": estimate_tokens(text)}
    per_skill = {
        skill: sum(v["estimated_tokens"] for k, v in files.items() if k.startswith(f"skills/{skill}/"))
        for skill in composed.skills
    }
    return {
        "loading": (
            "static: compose_report_profile() concatenates AGENTS.md, every shared reference, "
            "every declared SKILL.md and every skill reference into the prompt of every report "
            "run; the runtime's own skill tool is denied, so no skill is chosen or loaded on demand"
        ),
        "skills": list(composed.skills),
        "estimated_tokens_by_skill": per_skill,
        "files": files,
        "estimated_tokens_total": estimate_tokens(composed.instructions),
        "instructions_sha256": composed.content_sha256,
    }


def _runtime_invariants() -> list[str]:
    from toxagent.harness.context import SCIENTIFIC_INVARIANTS

    lines: list[str] = []
    for raw in SCIENTIFIC_INVARIANTS.replace("\\\n", "").splitlines():
        if raw.startswith("- "):
            lines.append(raw[2:].strip())
        elif lines and raw.strip():
            lines[-1] += " " + raw.strip()
    return lines


def build_matrix() -> dict[str, Any]:
    from toxagent.domain import explainer_validation

    models = _predictor_models()
    return {
        "schema_version": SCHEMA_VERSION,
        "predictor_models": models,
        "endpoints": _endpoints(models),
        "explainer_validation": explainer_validation.matrix(),
        "runtime_invariants": _runtime_invariants(),
        "tool_profiles": _tool_profiles(),
        "rollout_flags": _flags(),
        "instruction_surfaces": {
            "decision_support": _decision_support_prompt(),
            "report_build": _report_build_skills(),
        },
    }


def render_json(matrix: dict[str, Any]) -> str:
    return json.dumps(matrix, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(f"`{v}`" for v in value) or "—"
    return str(value).replace("|", "\\|")


def render_markdown(matrix: dict[str, Any]) -> str:
    out = [
        "# Capability matrix",
        "",
        "Generated by `python -m evals.capability_matrix --write` from the predictor registry",
        "manifests, `toxagent.flags`, `tools/registry.py`, the agent profile files and",
        "`domain/explainer_validation.py`. Do not edit by hand:",
        "`tests/unit/test_capability_matrix.py` fails when this file and the code disagree.",
        "It states what a deployment with **default flags** offers; anything behind a flag is",
        "listed as such, not as a capability.",
        "",
        "## Predictor models",
        "",
        "| Model | Endpoints | Status | Why blocked |",
        "|---|---|---|---|",
    ]
    for m in matrix["predictor_models"]:
        out.append(
            f"| `{m['model_id']}` | {_cell(m['endpoints'])} | {m['status']} | {_cell(m['blocked_reason'])} |"
        )
    out += ["", "## Endpoints", "", "| Endpoint | Status | Model |", "|---|---|---|"]
    for name, e in matrix["endpoints"].items():
        out.append(f"| `{name}` | {e['status']} | `{e['model_id']}` |")
    out += [
        "", "## Explainer validation", "",
        "What the served attribution method was measured to do. Every attribution a model reads",
        "carries the qualitative verdict (`explainer_validation` in the tool's model view).",
        "",
        "| Model | Target | Method | Determinism | Spelling invariance | Faithfulness wins vs random | Verdict | Measured |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in matrix["explainer_validation"]:
        target = x["endpoint"] + (f"/{x['task']}" if x["task"] else "")
        out.append(
            f"| `{x['model_id']}` | {target} | `{x['method']}` | {x['determinism']} | "
            f"{x['spelling_invariance']} | {x['faithfulness_wins']} | "
            f"`{x['faithfulness_vs_random_control']}` | {x['measured_on']} |"
        )
    out += ["", "Any other target, or any other method, is `not_measured`.", ""]
    out += ["## Scientific invariants stated to every runtime turn", ""]
    out += [f"- {line}" for line in matrix["runtime_invariants"]]
    out += [
        "", "## Tool profiles", "",
        "| Profile | Tools with default flags | Added only by a flag |",
        "|---|---|---|",
    ]
    for name, p in matrix["tool_profiles"].items():
        gated = ", ".join(f"`{t}` ({flag})" for t, flag in p["flag_gated"].items()) or "—"
        out.append(f"| `{name}` | {_cell(p['default'])} | {gated} |")
    out += [
        "", "## Rollout flags", "",
        "| Flag | Default | Owner | Remove by | Gates tools |",
        "|---|---|---|---|---|",
    ]
    for name, f in matrix["rollout_flags"].items():
        out.append(
            f"| `{name}` | {_cell(f['default'])} | {f['owner']} | {f['remove_by']} | {_cell(f['gates_tools'])} |"
        )
    ds = matrix["instruction_surfaces"]["decision_support"]
    rb = matrix["instruction_surfaces"]["report_build"]
    out += [
        "", "## Instruction surfaces", "",
        "Token counts are the deterministic estimate of `harness/prompt_budget.py`, never a",
        "provider count.", "",
        f"**decision_support** — {ds['loading']}; about {ds['estimated_tokens_total']} tokens "
        "before any session context.", "",
        "| Component | Estimated tokens |", "|---|---|",
    ]
    for name, c in ds["components"].items():
        out.append(f"| `{name}` | {c['estimated_tokens']} |")
    out += [
        "",
        f"**report_build** — {rb['loading']}. About {rb['estimated_tokens_total']} tokens per "
        "report run.", "",
        "| Skill | Estimated tokens (SKILL.md + references) |", "|---|---|",
    ]
    for skill, tokens in rb["estimated_tokens_by_skill"].items():
        out.append(f"| `{skill}` | {tokens} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    matrix = build_matrix()
    outputs = {JSON_PATH: render_json(matrix), MARKDOWN_PATH: render_markdown(matrix)}
    if args.write:
        for path, text in outputs.items():
            path.write_text(text)
        return 0
    if args.check:
        stale = [p for p, t in outputs.items() if not p.exists() or p.read_text() != t]
        for path in stale:
            print(
                f"{path.relative_to(REPO_ROOT)} is stale; run "
                "`python -m evals.capability_matrix --write` and review the diff",
                file=sys.stderr,
            )
        return 1 if stale else 0
    sys.stdout.write(outputs[JSON_PATH])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
