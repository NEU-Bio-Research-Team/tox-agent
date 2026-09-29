"""Experiment: OpenCode's native ``skill()`` on an isolated profile (RETHINK §4.9, W9-14).

RETHINK §4.9 proposes a quick check before relying on closed MCP tools for
skills: "test OpenCode V1 native skill() on an isolated profile that allows
only the approved skills; measure discovery/loading and context cost. Check
the binary's exact behaviour, its skill search path and whether references
can be read. Do not enable every skill from HOME or the user's directories,
do not open read/shell broadly." The production design (closed
``read_scientific_skill`` / ``read_skill_reference`` MCP tools, ADR 0012) did
not wait for it, so this is an optional experiment, recorded as one.

What it does, with the pinned binary:

1. Builds a throwaway workspace whose ``.opencode/skills/`` holds only the
   active decision-support skills of the shipped catalog (hash-pinned), and an
   agent ``skill-probe`` that denies every tool except ``skill``, and allows
   ``skill`` only for those names. ``read`` stays denied, as in production.
2. Starts ``opencode serve`` with its own HOME/XDG directories (credentials
   read from the runtime's isolated auth directory), external skill scans
   disabled, so nothing from the user's home can appear.
3. Records what the binary discovers (``GET /api/skill``), then sends a fixed
   set of prompts — one positive case per skill and negative controls — and
   records every ``skill`` call, whether it was allowed, attempted reference
   reads, tokens and latency.

It never touches the product database or the running control plane. Results
go to ``evals/experiments/runs/opencode-native-skills-<stamp>/``.

    python -m evals.experiments.opencode_native_skills [--model openai/gpt-5.6-luna]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toxagent.application.investigation.skill_catalog import load_catalog  # noqa: E402
from toxagent.platform.config import PACKAGE_ROOT  # noqa: E402
from toxagent.domain.provenance import content_sha256  # noqa: E402

PIN = "1.17.11"
AGENT = "skill-probe"
OUT = ROOT / "evals" / "experiments" / "runs"

#: One positive prompt per skill and two controls. Written before any run and
#: not tuned afterwards: the point is whether discovery works as described,
#: not to find prompts that make it look good.
PROMPTS: tuple[dict[str, Any], ...] = (
    {"id": "pos-conflict", "expect": "assess-conflicting-evidence",
     "text": "For cisapride, a colleague's radioligand binding screen showed only weak hERG "
             "displacement, while a published patch-clamp study reports potent hERG block. "
             "Which result should drive our decision, and why?"},
    {"id": "pos-attribution", "expect": "interpret-model-attribution",
     "text": "Our toxicity model's attribution highlights the basic amine of dofetilide as "
             "driving its hERG score. Does that tell us how dofetilide blocks the channel?"},
    {"id": "pos-critique", "expect": "critique-case",
     "text": "Compound A has an in-house hERG IC50 of 30 µM and a free Cmax of 50 nM at the "
             "intended dose. Should we prioritise it for the next study, and what would you "
             "test next?"},
    {"id": "neg-lookup", "expect": None,
     "text": "What does the abbreviation hERG stand for? One line."},
    {"id": "neg-arithmetic", "expect": None,
     "text": "Round 0.73064 to three decimals. Reply with the number only."},
)


#: Three arms on the same prompts. ``native`` is OpenCode's own behaviour;
#: ``instructed`` adds the one sentence the product's dynamic arm puts before
#: its index (``skill_catalog.render_index``), without the list, which the
#: binary supplies; ``denied`` turns ``skill`` off so the difference in input
#: tokens is what the native listing costs.
ARMS = ("native", "instructed", "denied")
INSTRUCTION = (
    "Each skill listed for the skill tool is a reviewed method for one kind of situation. "
    "When the situation in its description matches this question, load it with the skill "
    "tool before answering that part. Do not load skills that do not apply."
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _workspace(root: Path, arm: str = "native") -> tuple[Path, Path, list[dict[str, Any]]]:
    """The isolated project, its config file, and the pins of the skills it holds."""
    catalog = load_catalog(PACKAGE_ROOT / "agent_profiles")
    skills = [s for s in catalog.skills
              if s.status == "active" and "decision_support" in s.allowed_profiles]
    project = root / "project"
    for skill in skills:
        target = project / ".opencode" / "skills" / skill.skill_id
        source = next(
            (PACKAGE_ROOT / "agent_profiles" / d / skill.skill_id
             for d in ("scientific_skills",) if (PACKAGE_ROOT / "agent_profiles" / d / skill.skill_id).is_dir())
        )
        shutil.copytree(source, target)
        (target / "skill.manifest.json").unlink()  # ToxAgent metadata, not Agent Skills
    allow: Any = (
        "deny" if arm == "denied" else {"*": "deny", **{s.skill_id: "allow" for s in skills}}
    )
    config = {
        "$schema": "https://opencode.ai/config.json",
        # Declared, not left to discovery: a first run found that the binary
        # scans ``.opencode/skills`` only inside a detected project, and a
        # directory that is not a git worktree is the "global" project rooted
        # at ``/`` — the per-run workspaces of the product are exactly that.
        "skills": {"paths": [str(project / ".opencode" / "skills")]},
        "agent": {
            AGENT: {
                "description": "Isolated probe of native skill discovery (W9-14).",
                "mode": "primary",
                "permission": {
                    "*": "deny", "read": "deny", "edit": "deny", "glob": "deny", "grep": "deny",
                    "list": "deny", "bash": "deny", "task": "deny", "webfetch": "deny",
                    "websearch": "deny", "skill": allow,
                },
            }
        },
    }
    config_path = root / "opencode.json"
    config_path.write_text(json.dumps(config, indent=2))
    return project, config_path, [s.pin() for s in skills]


def _env(root: Path, config_path: Path) -> dict[str, str]:
    auth = REPO / ".data" / "opencode-auth" / "data"
    if not (auth / "opencode" / "auth.json").is_file():
        raise SystemExit(f"no isolated OpenCode credentials at {auth}; run ./bin/toxagent setup --agent")
    for name in ("home", "config", "state", "cache"):
        (root / name).mkdir(exist_ok=True)
    # The production runtime's model catalog, copied: a fresh cache fetched a
    # catalog without the deployed model on one run, and the experiment must
    # see the same models the product does.
    models = REPO / ".data" / "opencode-runtime" / "cache" / "opencode" / "models.json"
    if models.is_file():
        (root / "cache" / "opencode").mkdir(parents=True, exist_ok=True)
        shutil.copy2(models, root / "cache" / "opencode" / "models.json")
    return {
        "PATH": os.environ.get("PATH", ""), "HOME": str(root / "home"),
        "XDG_DATA_HOME": str(auth), "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_STATE_HOME": str(root / "state"), "XDG_CACHE_HOME": str(root / "cache"),
        "OPENCODE_CONFIG": str(config_path),
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1", "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        "TERM": "xterm",
    }


def _tool_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    parts = message.get("parts") or []
    return [p for p in parts if isinstance(p, dict) and p.get("type") == "tool"]


def _summarise_turn(prompt: dict[str, Any], message: dict[str, Any], seconds: float) -> dict[str, Any]:
    info = message.get("info") or {}
    tools = _tool_parts(message)
    skill_calls = []
    other_calls = []
    for part in tools:
        state = part.get("state") or {}
        entry = {"tool": part.get("tool"), "status": state.get("status"),
                 "input": state.get("input"), "error": state.get("error")}
        (skill_calls if part.get("tool") == "skill" else other_calls).append(entry)
    loaded = [((c.get("input") or {}).get("name")) for c in skill_calls
              if c.get("status") == "completed"]
    return {
        "prompt_id": prompt["id"], "expected_skill": prompt["expect"],
        "skill_calls": skill_calls, "skills_loaded": loaded, "other_tool_calls": other_calls,
        "hit": (prompt["expect"] in loaded) if prompt["expect"] else (not loaded),
        "tokens": info.get("tokens"), "model": info.get("modelID"), "provider": info.get("providerID"),
        "seconds": round(seconds, 1),
        "text": "".join(p.get("text", "") for p in message.get("parts") or []
                        if isinstance(p, dict) and p.get("type") == "text")[:2000],
    }


def _run_arm(arm: str, *, binary: Path, provider_id: str, model_id: str, out: Path,
             keep: bool) -> dict[str, Any]:
    scratch = Path(tempfile.mkdtemp(prefix=f"ocskill-{arm}-"))
    project, config_path, pins = _workspace(scratch, arm)
    port = _free_port()
    log = (out / f"opencode-{arm}.log").open("w")
    server = subprocess.Popen(
        [str(binary), "serve", "--pure", "--print-logs", "--log-level", "WARN",
         "--hostname", "127.0.0.1", "--port", str(port)],
        cwd=project, env=_env(scratch, config_path), stdout=log, stderr=subprocess.STDOUT,
    )
    config = json.loads(config_path.read_text())
    record: dict[str, Any] = {"arm": arm, "config": config, "config_sha256": content_sha256(config),
                              "catalog_pins": pins,
                              "system": INSTRUCTION if arm == "instructed" else None}
    try:
        query = {"directory": str(project)}
        with httpx.Client(base_url=f"http://127.0.0.1:{port}",
                          timeout=httpx.Timeout(600, connect=5)) as client:
            for _ in range(60):
                try:
                    if client.get("/agent", params=query).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise SystemExit(f"the isolated OpenCode server ({arm}) did not start")
            record["agent_visible"] = any(
                a.get("name") == AGENT for a in client.get("/agent", params=query).json()
            )
            turns = []
            for prompt in PROMPTS:
                session = client.post("/session", params=query, json={"title": prompt["id"]}).json()
                body: dict[str, Any] = {
                    "agent": AGENT, "model": {"providerID": provider_id, "modelID": model_id},
                    "parts": [{"type": "text", "text": prompt["text"]}],
                }
                if record["system"]:
                    body["system"] = record["system"]
                started = time.monotonic()
                response = client.post(f"/session/{session['id']}/message", params=query, json=body)
                elapsed = time.monotonic() - started
                raw = response.json() if response.headers.get("content-type", "").startswith(
                    "application/json") else {"status": response.status_code, "body": response.text}
                (out / f"raw-{arm}-{prompt['id']}.json").write_text(
                    json.dumps(raw, indent=2, default=str))
                turns.append(_summarise_turn(prompt, raw if isinstance(raw, dict) else {}, elapsed))
            record["turns"] = turns
            # Listings after the turns: a run that probed /experimental/tool
            # first had every turn fail with an internal server error, so the
            # probes must not be able to disturb what is being measured.
            # V1's own listing (``/api/skill`` is the v2 API and takes a
            # ``location`` object, which a first run got wrong).
            discovered = client.get("/skill", params=query)
            try:
                payload = discovered.json()
            except ValueError:
                payload = discovered.text
            items = payload.get("data", payload) if isinstance(payload, dict) else payload
            record["discovered_skills"] = sorted(
                (item.get("name") if isinstance(item, dict) else str(item))
                for item in (items if isinstance(items, list) else [])
            )
            tool_ids = client.get("/experimental/tool/ids", params=query)
            record["tool_ids"] = tool_ids.json() if tool_ids.status_code == 200 else tool_ids.status_code
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        if not keep:
            shutil.rmtree(scratch, ignore_errors=True)
    turns = record.get("turns", [])
    positives = [t for t in turns if t["expected_skill"]]
    negatives = [t for t in turns if not t["expected_skill"]]
    record["summary"] = {
        "answered": sum(1 for t in turns if t["tokens"]),
        "positive_hits": f"{sum(t['hit'] for t in positives)}/{len(positives)}",
        "negative_clean": f"{sum(t['hit'] for t in negatives)}/{len(negatives)}",
        "skill_calls": sum(len(t["skill_calls"]) for t in turns),
        "denied_skill_calls": sum(1 for t in turns for c in t["skill_calls"]
                                  if c.get("status") == "error"),
        "non_skill_tool_calls": sum(len(t["other_tool_calls"]) for t in turns),
        "input_tokens_on_negatives": [(t["tokens"] or {}).get("input") for t in negatives],
    }
    return record


def run(model: str, *, keep: bool = False, arms: tuple[str, ...] = ARMS) -> Path:
    binary = Path(os.environ.get("OPENCODE_BIN", Path.home() / ".opencode" / "bin" / "opencode"))
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
    if version != PIN:
        raise SystemExit(f"OpenCode {PIN} is pinned; found {version!r}")
    provider_id, _, model_id = model.partition("/")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT / f"opencode-native-skills-{stamp}"
    out.mkdir(parents=True)
    record: dict[str, Any] = {
        "schema_version": "opencode-native-skills-experiment-v2", "started_at": stamp,
        "binary_version": version, "model": model, "agent": AGENT, "prompts": list(PROMPTS),
        "arms": [
            _run_arm(arm, binary=binary, provider_id=provider_id, model_id=model_id, out=out,
                     keep=keep)
            for arm in arms
        ],
    }
    record["summary"] = {arm["arm"]: arm["summary"] for arm in record["arms"]}
    (out / "experiment.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("TOXAGENT_EXPERIMENT_MODEL",
                                                          "openai/gpt-5.6-luna"))
    parser.add_argument("--keep", action="store_true", help="keep the scratch workspace")
    args = parser.parse_args(argv)
    out = run(args.model, keep=args.keep)
    print(json.dumps(json.loads((out / "experiment.json").read_text())["summary"], indent=2))
    print(f"recorded in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
