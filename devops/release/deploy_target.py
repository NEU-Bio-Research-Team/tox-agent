"""Which deployment a git ref is allowed to change.

I30: the two auto-deploy workflows were written for a topology that no longer
exists. The backend one built `model_server/Dockerfile` and passed
`deploy/cloudrun-env.yaml`; neither path is in the tree. The frontend one ran
`firebase deploy --only hosting` with no `firebase.json` or `.firebaserc`
anywhere in the repository. Both would fail — but the more serious problem is
what they would have done had the paths still resolved: both triggered on
pushes to `agent_test`, and both deployed to the single live target. A test
branch published straight to production, with the runbook's staging service
existing only in the runbook.

Targets are resolved here rather than interpolated inside a shell step so the
mapping is one readable table, and so it can be asserted without a cloud
project — a workflow whose only test is running it against production is a
workflow nobody can change safely.

Rules, in the order they matter:

- Production is `main` and nothing else. An unrecognised ref resolves to
  nothing at all rather than to a default.
- Staging is `agent_test`. It is a different Cloud Run service, in the same
  project, with its own database and its own URL.
- Production does not build. It promotes an image digest that a staging
  deploy already published and smoke-tested, so what production runs is the
  artifact that was gated, not a rebuild of the same source that might differ.

Nothing here deploys, and nothing here contacts Google. It answers one
question — given a ref and a service, what is the target — and refuses when
there isn't one.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

#: The services this repository actually builds, and the Dockerfile plus build
#: context each one needs. The control image's context is its own directory on
#: purpose (ADR 0001): it must not be able to import the predictor even by
#: accident.
SERVICES: dict[str, dict[str, str]] = {
    "control": {
        "dockerfile": "backend/control/deploy/Dockerfile",
        "context": "backend/control",
        "cloud_run": "toxagent-control",
    },
    "predictor": {
        "dockerfile": "backend/predictor/deploy/Dockerfile",
        "context": ".",
        "cloud_run": "toxpred",
    },
    "ocr": {
        "dockerfile": "backend/ocr/deploy/Dockerfile",
        "context": ".",
        "cloud_run": "toxocr",
    },
    "frontend": {
        "dockerfile": "frontend/deploy/Dockerfile",
        "context": "frontend",
        "cloud_run": "toxagent-frontend",
    },
}

#: ref -> (environment, Cloud Run service suffix, builds or promotes).
ENVIRONMENTS: dict[str, dict[str, object]] = {
    "refs/heads/agent_test": {
        "environment": "staging",
        "suffix": "-staging",
        "builds": True,
    },
    "refs/heads/main": {
        "environment": "production",
        # Production promotes what staging gated. Building here would deploy an
        # artifact nothing has run.
        "suffix": "",
        "builds": False,
    },
}


class NoTarget(ValueError):
    """This ref does not deploy anywhere, and must not be given a default."""


@dataclass(frozen=True)
class Target:
    ref: str
    environment: str
    service: str
    cloud_run_service: str
    dockerfile: str
    context: str
    builds: bool


def resolve(ref: str, service: str) -> Target:
    if service not in SERVICES:
        raise NoTarget(f"unknown service {service!r}; this repository builds {sorted(SERVICES)}")
    rule = ENVIRONMENTS.get(ref)
    if rule is None:
        raise NoTarget(
            f"{ref!r} does not deploy; deploying refs are {sorted(ENVIRONMENTS)}"
        )
    definition = SERVICES[service]
    return Target(
        ref=ref,
        environment=str(rule["environment"]),
        service=service,
        cloud_run_service=f"{definition['cloud_run']}{rule['suffix']}",
        dockerfile=definition["dockerfile"],
        context=definition["context"],
        builds=bool(rule["builds"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", required=True, help="the full git ref, e.g. refs/heads/main")
    parser.add_argument("--service", required=True, choices=sorted(SERVICES))
    parser.add_argument(
        "--github-output", action="store_true",
        help="also write key=value lines to $GITHUB_OUTPUT",
    )
    args = parser.parse_args(argv)
    try:
        target = resolve(args.ref, args.service)
    except NoTarget as exc:
        print(f"refusing to deploy: {exc}", file=sys.stderr)
        return 2
    fields = asdict(target)
    print(json.dumps(fields, indent=2, sort_keys=True))
    if args.github_output:
        import os

        path = os.environ.get("GITHUB_OUTPUT")
        if path:
            with open(path, "a", encoding="utf-8") as handle:
                for key, value in fields.items():
                    handle.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
