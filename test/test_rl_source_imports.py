"""Static checks that the migrated RL modules target musa_titan."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = REPO_ROOT / "musa_titan" / "rl"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def test_rl_python_sources_do_not_import_musa_patch():
    py_files = list(RL_ROOT.rglob("*.py"))
    assert py_files
    leftover = {
        str(path.relative_to(REPO_ROOT)): sorted(
            name for name in _imported_modules(path) if name == "musa_patch"
        )
        for path in py_files
        if "musa_patch" in _imported_modules(path)
    }
    assert leftover == {}


def test_rl_python_sources_import_musa_titan_helpers():
    imported = {
        str(path.relative_to(REPO_ROOT)): sorted(_imported_modules(path))
        for path in RL_ROOT.rglob("*.py")
    }
    assert "musa_titan" in imported["musa_titan/rl/__init__.py"]
    assert "musa_titan" in imported["musa_titan/rl/slime_musa/__init__.py"]
    # Miles setup resolves the root package dynamically through importlib so
    # its initializer can remain independent of the shared RL helper module.
    assert "importlib" in imported["musa_titan/rl/miles_musa/__init__.py"]
