#!/usr/bin/env python3
"""Fail loudly if the live OpenCode agent surface is not deny-all-but-ToxAgent.

Progress log §4.2: even under ``serve --pure`` with ``OPENCODE_CONFIG`` pointed
at the profile, the ``toxagent`` agent inherited ``read: allow``, extra MCP
namespaces (``officecli``, ``codegraph``) and ``external_directory: allow`` from
the machine's ``~/.opencode``. ``test_opencode_profile.py`` only reads the
checked-in JSON, so it could not see this.

This reads the *running server's* ``GET /agent`` and checks the resolved
permission surface. ``run_local_phase3.sh`` calls it right after the server
comes up; the contract suite calls the same ``evaluate_surface`` against
captured payloads.

OpenCode 1.17.11's ``GET /agent`` reports ``permission`` as an **ordered list**
of ``{permission, pattern, action}`` rules, not the flat dict the checked-in
profile is written as — OpenCode's own built-in defaults for the agent's
*mode* (e.g. ``build``'s ``read: allow``) are prepended, and the profile's own
rules are appended after them. For a given ``(permission, pattern)`` pair the
*last* entry in the list is what is actually in effect (captured live: our
profile's ``read: deny`` after the base agent's ``read: allow`` for the same
``"*"`` pattern does end up denied) — this only resolves entries at the
wildcard ``"*"`` pattern, which is what a normal tool call matches; it does not
attempt full glob-priority resolution for narrower patterns such as
``*.env``. A flat dict (as the checked-in profile, or a hand-built test
payload) is accepted too.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

#: Built-in capabilities that must never resolve to ``allow`` for the product
#: agent (plan §11.2 deny-all list). ``external_directory: allow`` at the
#: wildcard pattern would mean unrestricted filesystem escape, not the narrow
#: per-path grants OpenCode makes for its own tool-output cache (those live at
#: non-``"*"`` patterns and are filtered out before this check ever sees them).
FORBIDDEN_BUILTINS = frozenset(
    {
        "read", "edit", "write", "glob", "grep", "list", "bash", "shell",
        "task", "subagent", "skill", "webfetch", "websearch", "execute",
        "patch", "external_directory",
    }
)

#: OpenCode's own interactive-UI permission types. Not in plan §11.2's deny-all
#: list — they gate a chat-mode interaction (asking the user a question,
#: entering plan mode), not tool/filesystem/network access — so a captured
#: agent that happens to resolve one to "allow" is not a leaked capability and
#: is not flagged.
UI_INTERACTION_PERMISSIONS = frozenset({"doom_loop", "question", "plan_enter", "plan_exit"})


def _is_own_mcp(key: str, agent_name: str) -> bool:
    """A permission key that names this product's own MCP tools."""
    return (
        key == agent_name
        or key.startswith(f"{agent_name}_")
        or key.startswith(f"{agent_name}*")
        or key.startswith(f"mcp_{agent_name}_")
        or key.startswith(f"mcp_{agent_name}*")
    )


def _resolve_star_pattern_rules(permission: object) -> dict[str, str] | None:
    """Reduce either shape of ``permission`` to ``{name: action}`` for the
    wildcard ``"*"`` pattern only, keeping the last entry per name in list
    order (OpenCode layers its own agent-mode defaults before the profile's
    own rules, so "last wins" is what the running server actually enforces)."""
    if isinstance(permission, dict):
        return dict(permission)
    if isinstance(permission, list):
        resolved: dict[str, str] = {}
        for rule in permission:
            if not isinstance(rule, dict):
                continue
            if rule.get("pattern") not in ("*", None):
                continue  # a narrower pattern; not what a plain tool call matches
            name, action = rule.get("permission"), rule.get("action")
            if isinstance(name, str) and isinstance(action, str):
                resolved[name] = action  # last write wins, matching list order
        return resolved
    return None


def evaluate_surface(agents: list[dict], agent_name: str) -> list[str]:
    """Return a list of human-readable problems; empty means the surface is
    deny-all except this product's own MCP namespace."""
    problems: list[str] = []
    match = next(
        (a for a in agents if isinstance(a, dict) and a.get("name") == agent_name), None
    )
    if match is None:
        return [f"agent {agent_name!r} is not present in GET /agent"]

    if match.get("mode") not in ("primary", None):
        problems.append(f"agent mode is {match.get('mode')!r}, expected 'primary'")

    permission = _resolve_star_pattern_rules(match.get("permission"))
    if permission is None:
        return [f"agent {agent_name!r} has no resolved permission list/map"]

    star = permission.get("*")
    if star not in ("deny", "ask"):
        problems.append(f'default permission "*" is {star!r}, expected "deny"')

    for key, effect in permission.items():
        if key == "*" or key in UI_INTERACTION_PERMISSIONS:
            continue
        if _is_own_mcp(key, agent_name):
            continue
        if effect != "allow":
            continue
        base = key.split("_", 1)[0].split("*", 1)[0]
        if base in FORBIDDEN_BUILTINS:
            problems.append(f'builtin {base!r} resolves to "allow" via permission {key!r}')
        else:
            # An unrecognised namespace resolving to allow at the wildcard
            # pattern is exactly what a leaked foreign MCP server (officecli,
            # codegraph) would look like.
            problems.append(f'unexpected permission {key!r} resolves to "allow"')

    # An allow rule for the product's own MCP must actually be present, or the
    # model is handed no tools at all (progress log §3.2).
    if not any(_is_own_mcp(k, agent_name) and e == "allow" for k, e in permission.items()):
        problems.append(
            f"no allow rule for the {agent_name!r} MCP namespace; the model would see no tools"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("TOXAGENT_OPENCODE_URL", "http://127.0.0.1:4096"))
    parser.add_argument("--agent", default=os.getenv("TOXAGENT_AGENT_NAME", "toxagent"))
    parser.add_argument("--directory", default=os.getenv("TOXAGENT_OPENCODE_DIRECTORY", ""))
    args = parser.parse_args()

    query = f"?directory={urllib.parse.quote(args.directory)}" if args.directory else ""
    url = args.url.rstrip("/") + "/agent" + query
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - localhost dev tool
            agents = json.loads(response.read())
    except (OSError, ValueError) as exc:
        print(f"could not read {url}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(agents, list):
        print(f"GET /agent returned {type(agents).__name__}, expected a list", file=sys.stderr)
        return 2

    problems = evaluate_surface(agents, args.agent)
    if problems:
        print(f"OpenCode agent {args.agent!r} surface is NOT isolated:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nRun OpenCode with an isolated HOME/XDG dir (see run_local_phase3.sh) "
            "so the machine's ~/.opencode and ~/.config/opencode do not leak in.",
            file=sys.stderr,
        )
        return 1
    print(f"OpenCode agent {args.agent!r} surface is deny-all except its own MCP namespace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
