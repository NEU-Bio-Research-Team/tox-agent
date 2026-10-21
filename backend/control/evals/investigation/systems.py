"""The systems of the comparison study, named once.

RETHINK §5.4 asks for four systems on the same cases — (A) predictor + template,
(B) an LLM given the predictor snapshot, (C) the current agent, (D) the
case-based investigator — and §5.5 for the skill ablation of D. The product
owner added general platforms as comparison arms (2026-09-25): OpenAI,
Anthropic and Google models, each bare and with the snapshot.

A ToxAgent system is a *deployment* (a control plane with a given flag set),
so its spec states the flags and skill arm the deployment must report through
``/v1/system/effective-product``; the runner refuses to record a ToxAgent arm
against a deployment that does not match, instead of mislabelling it.

``label`` is for the study report only. It never reaches the blinded packet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

#: Protocol v2 (#50): the four arms of the main comparison, as committed,
#: diffable files. The v1 arms below stay as diagnostics of the code as it was.
PROTOCOL_V2_DIR = Path(__file__).resolve().parent / "arms" / "protocol-v2"


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    rethink_arm: str
    family: str
    adapter: str
    label: str
    uses_snapshot: bool = False
    platform: str | None = None
    required_flags: Mapping[str, bool] = field(default_factory=dict)
    skills_mode: str | None = None
    #: "v1" for the historical arms; "investigation-protocol-v2" for #50's.
    protocol: str = "v1"
    #: The named run budget the deployment must report (#46).
    budget_preset: str | None = None
    #: Effective-product fields every arm of the protocol must share.
    same_across_arms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id, "rethink_arm": self.rethink_arm, "family": self.family,
            "adapter": self.adapter, "label": self.label, "uses_snapshot": self.uses_snapshot,
            "platform": self.platform, "required_flags": dict(self.required_flags),
            "skills_mode": self.skills_mode, "protocol": self.protocol,
            "budget_preset": self.budget_preset,
            "same_across_arms": list(self.same_across_arms),
        }


_CASE_OFF = {"scientific_case_v1": False, "scientific_skills_v1": False,
             "claim_reviewer_v1": False}

SYSTEMS: dict[str, SystemSpec] = {spec.system_id: spec for spec in (
    SystemSpec("A_predictor_template", "A", "predictor", "predictor_template",
               "ToxPred output rendered by a fixed template; no language model"),
    SystemSpec("C_toxagent_current", "C", "toxagent", "toxagent",
               "ToxAgent decision support as shipped (no case, no skills)",
               required_flags=_CASE_OFF, skills_mode="off"),
    SystemSpec("D_toxagent_investigator", "D", "toxagent", "toxagent",
               "ToxAgent case-based investigator, skills loaded on demand",
               required_flags={"scientific_case_v1": True, "scientific_skills_v1": True,
                               "answer_draft_v2": True, "claim_reviewer_v1": False},
               skills_mode="dynamic"),
    SystemSpec("Dr_toxagent_investigator_reviewer", "D-ablation", "toxagent", "toxagent",
               "ToxAgent investigator (as D) plus the independent claim-support reviewer "
               "(W9-12); the reviewer's verdicts are recorded in the run's decision state and "
               "never change the answer, so it is graded against D on the same cases",
               required_flags={"scientific_case_v1": True, "scientific_skills_v1": True,
                               "answer_draft_v2": True, "claim_reviewer_v1": True},
               skills_mode="dynamic"),
    SystemSpec("D0_toxagent_case_no_skills", "D-ablation", "toxagent", "toxagent",
               "ToxAgent case-based investigator without skills",
               required_flags={"scientific_case_v1": True, "scientific_skills_v1": False,
                               "answer_draft_v2": True, "claim_reviewer_v1": False},
               skills_mode="off"),
    SystemSpec("Ds_toxagent_case_static_skills", "D-ablation", "toxagent", "toxagent",
               "ToxAgent case-based investigator, every skill composed into the prompt",
               required_flags={"scientific_case_v1": True, "scientific_skills_v1": False,
                               "answer_draft_v2": True, "claim_reviewer_v1": False},
               skills_mode="static"),
    SystemSpec("P_openai_bare", "P", "platform", "codex_cli",
               "OpenAI model through the codex CLI, question only", platform="openai"),
    SystemSpec("B_openai_snapshot", "B", "platform", "codex_cli",
               "OpenAI model through the codex CLI, with the ToxPred snapshot",
               uses_snapshot=True, platform="openai"),
    SystemSpec("P_anthropic_bare", "P", "platform", "claude_cli",
               "Anthropic model through the claude CLI (no tools), question only",
               platform="anthropic"),
    SystemSpec("B_anthropic_snapshot", "B", "platform", "claude_cli",
               "Anthropic model through the claude CLI (no tools), with the ToxPred snapshot",
               uses_snapshot=True, platform="anthropic"),
    SystemSpec("P_google_bare", "P", "platform", "google",
               "Google Gemini model through the operator's MCP bridge (or answered out of "
               "process with --google-channel manual), question only", platform="google"),
    SystemSpec("B_google_snapshot", "B", "platform", "google",
               "Google Gemini model through the operator's MCP bridge (or answered out of "
               "process with --google-channel manual), with the ToxPred snapshot",
               uses_snapshot=True, platform="google"),
)}


def arm_config(path: Path) -> dict[str, Any]:
    """One protocol arm, merged over what it extends: scalars replace,
    ``required_flags`` merge key by key. The merged form is what a diff
    between two arms reads."""
    config = json.loads(path.read_text())
    parent = config.pop("extends", None)
    if parent is None:
        return config
    base = arm_config(path.parent / parent)
    flags = {**base.get("required_flags", {}), **config.get("required_flags", {})}
    return {**base, **config, "required_flags": flags}


def protocol_arms(directory: Path = PROTOCOL_V2_DIR) -> dict[str, SystemSpec]:
    arms: dict[str, SystemSpec] = {}
    for path in sorted(directory.glob("[A-Z].json")):
        config = arm_config(path)
        spec = SystemSpec(
            system_id=config["system_id"], rethink_arm=config["arm"], family=config["family"],
            adapter=config["adapter"], label=config["label"],
            required_flags=dict(config.get("required_flags") or {}),
            skills_mode=config.get("skills_mode") if config["family"] == "toxagent" else None,
            protocol=config["protocol"], budget_preset=config.get("budget_preset"),
            same_across_arms=tuple(config.get("same_across_arms") or ()),
        )
        arms[spec.system_id] = spec
    return arms


def arm_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """What differs between two merged arm configs, flags compared one by one.
    Prose fields (about, label, role) and the arm's own name are not variables."""
    skip = {"about", "label", "role", "arm", "system_id"}
    out: dict[str, tuple[Any, Any]] = {}
    for key in sorted((set(left) | set(right)) - skip - {"required_flags"}):
        if left.get(key) != right.get(key):
            out[key] = (left.get(key), right.get(key))
    flags_l, flags_r = left.get("required_flags", {}), right.get("required_flags", {})
    for flag in sorted(set(flags_l) | set(flags_r)):
        if flags_l.get(flag) != flags_r.get(flag):
            out[f"required_flags.{flag}"] = (flags_l.get(flag), flags_r.get(flag))
    return out


def _field(product: Mapping[str, Any], path: str) -> Any:
    if path == "budget.preset":
        return (((product.get("intents") or {}).get("decision_support") or {})
                .get("budget") or {}).get("preset")
    value: Any = product
    for part in path.split("."):
        value = (value or {}).get(part) if isinstance(value, Mapping) else None
    return value


def fairness_problems(specs: list[SystemSpec],
                      products: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Why arms of one protocol are not comparable: a shared field -- the
    backbone, the budget, a pinned corpus or replay file -- that two of their
    deployments report differently. Empty when they agree."""
    problems: list[str] = []
    by_protocol: dict[str, list[SystemSpec]] = {}
    for spec in specs:
        if spec.same_across_arms and spec.system_id in products:
            by_protocol.setdefault(spec.protocol, []).append(spec)
    for protocol, group in by_protocol.items():
        for path in sorted({p for spec in group for p in spec.same_across_arms}):
            seen = {spec.system_id: _field(products[spec.system_id], path) for spec in group}
            if len({json.dumps(v, sort_keys=True, default=str) for v in seen.values()}) > 1:
                problems.append(f"{protocol}: {path} differs across arms: {seen}")
    return problems


SYSTEMS.update(protocol_arms())


def resolve(names: list[str]) -> list[SystemSpec]:
    unknown = sorted(set(names) - set(SYSTEMS))
    if unknown:
        raise ValueError(f"unknown systems {unknown}; known: {sorted(SYSTEMS)}")
    return [SYSTEMS[name] for name in names]


def product_mismatch(spec: SystemSpec, effective_product: Mapping) -> list[str]:
    """Why a deployment is not the arm it would be recorded as (empty if it is)."""
    problems: list[str] = []
    flags = effective_product.get("flags") or {}
    for name, wanted in spec.required_flags.items():
        if name not in flags and wanted is False:
            continue  # a deployment that predates a flag has it off
        actual = (flags.get(name) or {}).get("enabled")
        if actual is not wanted:
            problems.append(f"flag {name} is {actual}, the arm needs {wanted}")
    if spec.skills_mode is not None:
        actual_mode = (effective_product.get("scientific_skills") or {}).get("mode", "off")
        if actual_mode != spec.skills_mode:
            problems.append(f"skills mode is {actual_mode}, the arm needs {spec.skills_mode}")
    if spec.budget_preset is not None:
        actual_preset = _field(effective_product, "budget.preset")
        if actual_preset != spec.budget_preset:
            problems.append(
                f"budget preset is {actual_preset}, the arm needs {spec.budget_preset}"
            )
    return problems
