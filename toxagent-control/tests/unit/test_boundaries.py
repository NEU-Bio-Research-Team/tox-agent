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

PACKAGE = Path(__file__).resolve().parents[2] / "toxagent"

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
