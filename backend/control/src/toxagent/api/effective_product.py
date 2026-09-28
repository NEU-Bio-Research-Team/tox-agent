"""What product is this deployment actually running? (effective-product-v1)

Two benchmark runs with the same suite hash could be grading two different
products: rollout flags change which report path runs, which router decides,
which answer wire shape is accepted, and whether workers are in-process. A
manifest that recorded only a commit and a runtime name could not tell them
apart.

This module states the effective configuration once, from the same objects the
application composes itself from — the flag catalogue, the tool registry, the
runtime profile registry, the settings and the budget — so the eval runner (in
process) and a live stack (over ``GET /v1/system/effective-product``) produce
the same document. It never includes a secret: hosts, not URLs with
credentials; provider and model ids, not keys.
"""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .. import flags as rollout
from ..config import Settings
from ..domain.provenance import content_sha256
from ..domain.run import Intent
from ..application.run_budget import budget_matrix

SCHEMA_VERSION = "effective-product-v1"

#: Intents a new request can resolve to. The three ADR 0010 retired intents
#: are historical reads only and would misstate the live surface.
LIVE_INTENTS: tuple[str, ...] = tuple(
    intent.value
    for intent in Intent
    if intent not in (Intent.REPORT_QA, Intent.EVIDENCE_RESEARCH, Intent.ATTRIBUTION)
)

_DETERMINISTIC_INTENTS = frozenset(
    {
        Intent.ANALYSIS.value, Intent.ANALYSIS_BATCH.value,
        Intent.STRUCTURE_RECOGNITION.value,
        Intent.CLARIFICATION_REQUIRED.value, Intent.OUT_OF_SCOPE.value,
    }
)


def _host(url: str) -> str | None:
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    # hostname drops any userinfo, which is where a credential would sit.
    return parts.hostname or None


def flag_snapshot(today: date | None = None) -> dict[str, dict[str, Any]]:
    today = today or date.today()
    return {
        item.name: {
            "enabled": item.enabled(),
            "default": item.default,
            "overridden": item.enabled() != item.default,
            "remove_by": item.remove_by.isoformat(),
            "expired": today > item.remove_by,
        }
        for item in rollout.FLAGS
    }


def prompt_hashes(profiles_dir: Path | None) -> dict[str, str]:
    """Hashes of the static prompt text and every shipped profile file.

    The per-run system prompt is dynamic (it embeds the session), and each
    run already records its own hash. What a benchmark needs to compare is
    the static policy text and the deployed profile documents.
    """
    from ..harness import context

    static = {
        name: getattr(context, name)
        for name in sorted(dir(context))
        if name.isupper() and isinstance(getattr(context, name), str)
    }
    hashes = {"static_policy_text": content_sha256(static)}
    if profiles_dir is not None and profiles_dir.is_dir():
        digest = hashlib.sha256()
        for path in sorted(p for p in profiles_dir.rglob("*") if p.is_file()):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(profiles_dir).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
        hashes["agent_profiles"] = digest.hexdigest()
    return hashes


def intent_lane(intent: str, *, report_orchestrator_v2: bool) -> str:
    if intent in _DETERMINISTIC_INTENTS:
        return "deterministic"
    if intent == Intent.BUILD_REPORT.value and report_orchestrator_v2:
        return "orchestrated"
    return "agentic"


def _skills_section(settings: Settings, flags: dict[str, Any]) -> dict[str, Any]:
    from ..application.skill_catalog import load_catalog

    if flags["scientific_skills_v1"]["enabled"]:
        mode = "dynamic"
    elif getattr(settings.runtime, "scientific_skills_static", False):
        mode = "static"
    else:
        mode = "off"
    return {"mode": mode, **load_catalog(settings.profiles_dir).manifest()}


def describe_effective_product(
    settings: Settings,
    *,
    tool_registry=None,
    capabilities: dict[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """The effective product configuration, secret-free."""
    from .. import __version__
    from ..harness.runtime_profiles import registry_from_settings
    from ..tools.registry import PROFILE_MANIFEST, PROFILES, ToolRegistry

    flags = flag_snapshot(today)
    orchestrated = flags["report_orchestrator_v2"]["enabled"]
    runtime_profiles = registry_from_settings(
        settings.runtime, settings,
        select_named_agents=flags["runtime_profile_selector_v2"]["enabled"],
    )
    registry: ToolRegistry | None = tool_registry
    budgets = budget_matrix(settings.policy, settings.runtime)

    intents: dict[str, Any] = {}
    for intent in LIVE_INTENTS:
        lane = intent_lane(intent, report_orchestrator_v2=orchestrated)
        entry: dict[str, Any] = {"lane": lane, "budget": budgets.get(intent)}
        if lane != "deterministic":
            profile = (
                "report_synthesis" if lane == "orchestrated"
                else (registry.profile_for_intent(intent) if registry else
                      ToolRegistry().profile_for_intent(intent))
            )
            declared = sorted(PROFILES.get(profile, ()))
            entry["capability_profile"] = profile
            entry["declared_tools"] = declared
            if registry is not None:
                entry["registered_tools"] = [t.name for t in registry.visible_for(profile)]
                entry["tool_schema_hash"] = registry.schema_hash(profile)
            entry["runtime_binding"] = runtime_profiles.resolve(
                intent, capability_profile=profile
            ).to_manifest()
        intents[intent] = entry

    worker = settings.worker
    runtime = settings.runtime
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "toxagent_version": __version__,
        "flags": flags,
        "expired_flags": sorted(name for name, f in flags.items() if f["expired"]),
        "intents": intents,
        # W9-09: the permission manifest every profile above was read from.
        "tool_profiles": PROFILE_MANIFEST.summary(),
        "runtime": {
            "kind": runtime.kind,
            "provider_id": runtime.provider_id,
            "model_id": runtime.model_id,
            "runtime_version": (
                runtime.opencode_version if runtime.kind == "opencode"
                else runtime.dsh_version if runtime.kind == "dsh" else None
            ),
            "agent_name": runtime.agent_name,
            "report_agent_name": runtime.report_agent_name,
            # Per-session model connections can override provider credentials;
            # the deployment default is what an unconfigured session runs on.
            "auth_mode_default": "runtime_host",
        },
        "topology": {
            "external_worker_mode": flags["external_worker_mode"]["enabled"],
            "process_role": worker.role,
            "queues": list(worker.queues),
            "max_in_flight": worker.max_in_flight,
            "concurrency_caps": {
                "global": worker.global_max_runs, "tenant": worker.tenant_max_runs,
                "provider": worker.provider_max_runs, "report": worker.report_max_runs,
            },
            "max_run_attempts": worker.max_run_attempts,
            "max_concurrent_runs_per_session": settings.policy.max_concurrent_runs_per_session,
        },
        "providers": {
            "predictor_host": _host(settings.predictor.base_url),
            "research_provider": settings.research.provider or None,
            "research_host": _host(settings.research.base_url),
            "research_max_results": settings.research.max_results,
            # A benchmark on frozen evidence says so, and says which snapshot.
            "research_snapshot": (
                {"path_name": Path(settings.research.snapshot_path).name,
                 "fault": settings.research.snapshot_fault or None}
                if settings.research.provider == "snapshot" else None
            ),
            # A benchmark over a local corpus says so, and says which corpus:
            # the pin is what ties a recorded number to the abstracts it ranked
            # (research/providers/corpus.py).
            "research_corpus": (
                {"path_name": Path(settings.research.corpus_path).name,
                 "sha256": settings.research.corpus_sha256 or None}
                if settings.research.provider == "corpus" else None
            ),
            "compound_provider": getattr(settings.compound, "provider", None) or None,
            "ocr_configured": bool(settings.ocr.base_url),
        },
        # ADR 0012 / RETHINK §5.5: which skill arm this deployment runs and
        # the exact skill texts, so two runs of the ablation can be told apart.
        "scientific_skills": _skills_section(settings, flags),
        "hashes": {
            **prompt_hashes(settings.profiles_dir),
            **({"tool_registry": registry.schema_hash()} if registry is not None else {}),
        },
    }
    if capabilities is not None:
        document["capabilities"] = capabilities
    document["effective_product_hash"] = content_sha256(
        {k: v for k, v in document.items() if k != "capabilities"}
    )
    return document
