"""What a customer distribution contains, checked rather than remembered (K12).

`docs/spec/WORKSPACE_HANDOFF_SIMPLIFICATION_PLAN_VI.md` classifies the tree in
a table a person is expected to apply. A table a person applies goes stale the
first time somebody adds a file, and the two ways it goes wrong are not
symmetric: shipping `docs/audit/` discloses internal findings, and withholding
a file the setup procedure needs hands the customer a clone that cannot start.

So every tracked path must match exactly one rule in
`devops/handoff_allowlist.json`. A path matching neither is an error, not a
default — adding a file becomes a decision about whether it ships.

    handoff.py --check          # classify every tracked path; unmatched fails
    handoff.py --list           # the included paths, one per line
    handoff.py --stage <dir>    # copy the included paths into an empty dir

`--stage` produces a directory to review. It does not create a repository,
push anything, or rewrite history; the plan is explicit that a delivery repo
is a separate act, and this is the part that can be checked beforehand.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWLIST = ROOT / "devops" / "handoff_allowlist.json"

#: Credential shapes that must not travel in a distribution. `.env.example` is
#: a template with empty values and is expected to match none of these.
SECRET_SHAPES = (
    re.compile(r"\b(?:sk|rk|xoxb|ghp|gho|glpat)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9._-]{16,}\.[A-Za-z0-9._-]{8,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:/@]+:(?!password|changeme|toxagent\b)[^\s@]{8,}@"),
)

#: Files that are compiled, archived or otherwise not text worth scanning.
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".pptx", ".zip", ".gz",
    ".pt", ".safetensors", ".woff", ".woff2", ".ttf", ".onnx", ".npy",
}


def tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def rules() -> tuple[list[dict], list[dict]]:
    document = json.loads(ALLOWLIST.read_text())
    return document["include"], document["exclude"]


def secret_shape_exemptions() -> dict[str, str]:
    """Files whose credential-shaped content is deliberate.

    A test cannot prove a credential is redacted without containing one. Kept
    as data rather than an inline marker in the file, so adding an exemption
    is a reviewed change to a short list instead of a comment nobody sees.
    """
    document = json.loads(ALLOWLIST.read_text())
    return {entry["path"]: entry["why"] for entry in document.get("secret_shape_exemptions", [])}


def _matches(path: str, glob: str) -> bool:
    # `**` in these globs means "this directory and everything under it",
    # which fnmatch does not do on its own.
    if glob.endswith("/**"):
        return path == glob[:-3] or path.startswith(glob[:-2])
    return fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(Path(path).name, glob)


def classify(paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    include, exclude = rules()
    included, excluded, unmatched = [], [], []
    for path in paths:
        # Exclusions win: a path under an included directory that is also
        # named by an exclusion is withheld. The reverse would make every
        # exclusion depend on no broader include existing.
        if any(_matches(path, rule["glob"]) for rule in exclude):
            excluded.append(path)
        elif any(_matches(path, rule["glob"]) for rule in include):
            included.append(path)
        else:
            unmatched.append(path)
    return included, excluded, unmatched


def secrets_in(paths: list[str]) -> list[tuple[str, int]]:
    """Credential shapes in files that would be handed over."""
    exempt = secret_shape_exemptions()
    findings = []
    for path in paths:
        if path in exempt:
            continue
        full = ROOT / path
        if full.suffix.lower() in BINARY_SUFFIXES or not full.is_file():
            continue
        try:
            text = full.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if any(shape.search(line) for shape in SECRET_SHAPES):
                findings.append((path, number))
    return findings


#: Markdown links, ignoring images. Anchors and external schemes are handled
#: below rather than in the pattern, so the reason for each is visible.
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def dangling_references(included: list[str]) -> list[tuple[str, str]]:
    """Links from a shipped document to a path that is not shipped.

    The second failure mode, and the quieter one. Withholding an internal
    document is correct; leaving a link to it in a document the customer does
    receive hands them a dead link to something they are not allowed to have,
    which reads as a broken distribution rather than as a boundary.
    """
    shipped = set(included)
    findings = []
    for path in included:
        if not path.endswith(".md"):
            continue
        base = Path(path).parent
        for target in _LINK.findall((ROOT / path).read_text(encoding="utf-8")):
            target = target.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            resolved = (base / target).as_posix()
            resolved = Path(resolved).resolve().relative_to(ROOT).as_posix() if (
                (ROOT / base / target).exists()
            ) else resolved.lstrip("./")
            if resolved in shipped:
                continue
            # A link to a directory is satisfied by anything shipped under it.
            if any(candidate.startswith(resolved.rstrip("/") + "/") for candidate in shipped):
                continue
            findings.append((path, target))
    return findings


def check() -> int:
    included, excluded, unmatched = classify(tracked())
    problems = 0

    # An exemption for a file that no longer exists, or that is not shipped
    # anyway, is an exemption nobody will notice has stopped meaning anything.
    stale = [
        path for path in secret_shape_exemptions()
        if not (ROOT / path).is_file() or path not in set(included)
    ]
    if stale:
        problems += 1
        print("secret-shape exemptions that no longer apply:", file=sys.stderr)
        for path in stale:
            print(f"  - {path}", file=sys.stderr)

    if unmatched:
        problems += 1
        print(
            f"{len(unmatched)} tracked path(s) match no rule in "
            f"{ALLOWLIST.relative_to(ROOT)}; each has to be a decision:",
            file=sys.stderr,
        )
        for path in unmatched[:40]:
            print(f"  - {path}", file=sys.stderr)
        if len(unmatched) > 40:
            print(f"  ... and {len(unmatched) - 40} more", file=sys.stderr)

    dangling = dangling_references(included)
    if dangling:
        problems += 1
        print(
            "shipped documents linking to paths that are withheld — a customer "
            "clone would have dead links:",
            file=sys.stderr,
        )
        for path, target in dangling:
            print(f"  - {path} -> {target}", file=sys.stderr)

    findings = secrets_in(included)
    if findings:
        problems += 1
        print("credential-shaped content in files that would be handed over:", file=sys.stderr)
        for path, number in findings[:20]:
            print(f"  - {path}:{number}", file=sys.stderr)

    print(f"handoff: {len(included)} included, {len(excluded)} withheld, {len(unmatched)} undecided")
    return 1 if problems else 0


def stage(destination: Path) -> int:
    if destination.exists() and any(destination.iterdir()):
        print(f"{destination} is not empty", file=sys.stderr)
        return 1
    included, _, unmatched = classify(tracked())
    if unmatched:
        print("refusing to stage while paths are undecided; run --check", file=sys.stderr)
        return 1
    for path in included:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, target)
    print(f"staged {len(included)} files into {destination}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--list", action="store_true")
    group.add_argument("--stage", type=Path)
    args = parser.parse_args(argv)

    if args.check:
        return check()
    if args.list:
        for path in classify(tracked())[0]:
            print(path)
        return 0
    return stage(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
