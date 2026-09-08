"""Dependency rule enforcement (plan section 3.1).

    api -> application -> domain
                       -> scientific interfaces
    domain -> standard library only

Checked by static import inspection so a violation fails CI without needing the
heavy dependencies installed.
"""
import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "toxpred"

FORBIDDEN_EVERYWHERE = {
    "google", "google_adk", "google_genai", "firebase_admin", "firestore",
    "deepchem", "tdc", "sentence_transformers", "molscribe", "onmt",
    "agents", "services", "tools", "model_server", "src",
}

FORBIDDEN_IN_DOMAIN = FORBIDDEN_EVERYWHERE | {
    "torch", "rdkit", "transformers", "fastapi", "numpy", "pandas",
    "sklearn", "matplotlib", "yaml", "backend",
}


def module_files(subdir: str) -> list[Path]:
    return sorted((PACKAGE / subdir).rglob("*.py"))


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", module_files("domain"), ids=lambda p: p.name)
def test_domain_layer_is_pure(path):
    offenders = top_level_imports(path) & FORBIDDEN_IN_DOMAIN
    assert not offenders, f"{path.name} imports {sorted(offenders)}; domain must stay pure"


@pytest.mark.parametrize(
    "path",
    module_files("domain") + module_files("scientific") + module_files("application"),
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_no_agent_or_cloud_dependencies(path):
    offenders = top_level_imports(path) & FORBIDDEN_EVERYWHERE
    assert not offenders, f"{path.relative_to(PACKAGE)} imports {sorted(offenders)}"


def test_importing_the_package_does_not_load_a_model():
    """`import toxpred` must not read a checkpoint or touch the network."""
    import subprocess
    import sys

    code = (
        "import sys; import toxpred; "
        "loaded = {m for m in sys.modules if m.split('.')[0] in "
        "{'torch','transformers','rdkit'}}; "
        "assert not loaded, loaded; print('clean')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PACKAGE.parent, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
