"""Dependency rules (plan section 4.1, ADR 0001).

    api -> application -> domain
    application -> predictor client, research interfaces, persistence interfaces
    harness adapters -> runtime provider interface
    tool gateway -> application services, never the runtime gateway
    domain -> standard library only

Checked by parsing imports rather than by importing, so a violation fails
without needing the heavy dependencies installed — and so the failure names the
file and the offending import instead of a traceback from three layers down.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import toxagent

# Through the package, not by counting directories up from this file. The
# src-layout move left the old count pointing at backend/control/toxagent,
# which does not exist, so every rglob below returned nothing, every
# parametrize was empty, and the four tests that enforce ADR 0001 reported
# "skipped" while asserting against zero files. pyproject.toml now sets
# empty_parameter_set_mark = fail_at_collect so a repeat is a red build.
PACKAGE = Path(toxagent.__file__).resolve().parent

# The predictor lives behind an HTTP contract. Importing it here would make the
# control plane un-deployable without model artifacts and would let scientific
# semantics leak in through Python instead of through the versioned schema.
FORBIDDEN_EVERYWHERE = {
    "toxpred", "backend", "torch", "transformers", "rdkit", "deepchem", "tdc",
    "numpy", "pandas", "sklearn", "google", "firebase_admin",
}

FORBIDDEN_IN_DOMAIN = FORBIDDEN_EVERYWHERE | {
    "fastapi", "starlette", "sqlalchemy", "httpx", "mcp", "jwt", "alembic", "pydantic",
}


def modules(subdir: str) -> list[Path]:
    return sorted((PACKAGE / subdir).rglob("*.py"))


def imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def relative_targets(path: Path) -> set[str]:
    """Sibling packages reached by relative import, as top-level package names."""
    tree = ast.parse(path.read_text())
    package_parts = path.relative_to(PACKAGE).parts[:-1]
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = list(package_parts[: len(package_parts) - (node.level - 1)])
            parts = base + ((node.module or "").split(".") if node.module else [])
            if parts:
                targets.add(parts[0])
    return targets


@pytest.mark.parametrize("path", modules("domain"), ids=lambda p: p.name)
def test_domain_is_pure_python(path):
    offenders = imports(path) & FORBIDDEN_IN_DOMAIN
    assert not offenders, f"{path.name} imports {sorted(offenders)}; domain stays stdlib-only"


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_nothing_imports_the_predictor_or_a_model(path):
    offenders = imports(path) & FORBIDDEN_EVERYWHERE
    assert not offenders, (
        f"{path.relative_to(PACKAGE)} imports {sorted(offenders)}; the predictor is reached "
        "over its versioned HTTP contract (ADR 0001)"
    )


@pytest.mark.parametrize("path", modules("domain"), ids=lambda p: p.name)
def test_domain_does_not_depend_on_outer_layers(path):
    forbidden = {"api", "application", "persistence", "tools", "harness", "research", "predictor"}
    offenders = relative_targets(path) & forbidden
    assert not offenders, f"{path.name} depends on {sorted(offenders)}; domain is the innermost layer"


@pytest.mark.parametrize("path", modules("tools"), ids=lambda p: p.name)
def test_tool_gateway_does_not_call_the_runtime_gateway(path):
    """Plan section 4.1. Tools serve the runtime; a tool that could start a
    runtime turn would let a model recurse into itself through the tool plane."""
    assert "harness" not in relative_targets(path), (
        f"{path.name} imports the harness; tools run beneath the runtime, not beside it"
    )


# -- layer order ------------------------------------------------------------
#
# Outermost first. A module may import its own package, any package on a lower
# line, and nothing else: not a higher line, and not a *different* package on
# its own line. Every top-level package or module must appear here, so a new
# one is a placement decision, not a default. The reasoning is in
# docs/spec/WORKSPACE_STRUCTURE_REVIEW.md, section 6.2.
LAYERS: tuple[tuple[str, ...], ...] = (
    ("worker",),
    ("api",),
    ("harness",),
    ("tools",),
    ("application",),
    ("superseded",),                # the ADR 0011 kernel, kept until its tables retire
    ("report",),
    ("validation",),
    ("persistence",),
    ("research", "predictor", "connections", "streaming"),
    ("domain", "platform"),
)

#: Imports that break the order today, each with the reason it has not moved
#: yet. The list may only shrink: ``test_every_layer_exception_is_still_real``
#: fails once an entry stops being true, so a fix has to delete its line.
LAYER_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("persistence/investigations.py", "superseded"): "the kernel's own store (ADR 0011)",
    ("persistence/sql/repositories/investigation.py", "superseded"): "the kernel's own store (ADR 0011)",
}

_LEVEL = {name: level for level, line in enumerate(LAYERS) for name in line}


def _top_level_names() -> set[str]:
    return {
        p.stem if p.is_file() else p.name
        for p in PACKAGE.iterdir()
        if (p.is_file() and p.suffix == ".py" and p.stem != "__init__")
        or (p.is_dir() and (p / "__init__.py").exists())
    }


def _package_targets(path: Path) -> set[str]:
    """Top-level ``toxagent`` names this file imports, relative or absolute."""
    targets = relative_targets(path)
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            parts = node.module.split(".")
        elif isinstance(node, ast.Import):
            parts = [p for alias in node.names for p in alias.name.split(".")[:2]]
        else:
            continue
        if parts[:1] == ["toxagent"] and len(parts) > 1:
            targets.add(parts[1])
    return {t for t in targets if t in _LEVEL}


def _layer_violations(path: Path) -> set[str]:
    rel = path.relative_to(PACKAGE)
    source = rel.parts[0].removesuffix(".py")
    if source == "__init__":
        return set()
    here = _LEVEL[source]
    bad = set()
    for target in _package_targets(path) - {source}:
        there = _LEVEL[target]
        if there <= here:
            bad.add(target)
    return bad


def test_every_package_has_a_layer():
    unplaced = _top_level_names() - set(_LEVEL)
    assert not unplaced, f"place {sorted(unplaced)} in LAYERS"
    stale = set(_LEVEL) - _top_level_names()
    assert not stale, f"LAYERS names packages that no longer exist: {sorted(stale)}"


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_imports_follow_the_layer_order(path):
    rel = path.relative_to(PACKAGE).as_posix()
    offenders = {t for t in _layer_violations(path) if (rel, t) not in LAYER_EXCEPTIONS}
    assert not offenders, (
        f"{rel} imports {sorted(offenders)}, which sit above or beside it in LAYERS; "
        "move the shared piece down a layer instead of importing up"
    )


@pytest.mark.parametrize("key", sorted(LAYER_EXCEPTIONS), ids=lambda k: f"{k[0]}->{k[1]}")
def test_every_layer_exception_is_still_real(key):
    rel, target = key
    assert target in _layer_violations(PACKAGE / rel), (
        f"{rel} no longer imports {target}; delete its LAYER_EXCEPTIONS entry"
    )
