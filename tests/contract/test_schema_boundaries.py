"""Schema boundary contracts for component envelopes."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOTS = (
    REPO_ROOT / "mdp" / "cli" / "train.py",
    REPO_ROOT / "mdp" / "cli" / "rl_train.py",
    REPO_ROOT / "mdp" / "cli" / "estimate.py",
    REPO_ROOT / "mdp" / "assembly",
    REPO_ROOT / "mdp" / "data",
    REPO_ROOT / "mdp" / "training",
    REPO_ROOT / "mdp" / "serving",
    REPO_ROOT / "mdp" / "utils" / "estimator.py",
)
SERIALIZATION_HELPERS = {
    REPO_ROOT / "mdp" / "assembly" / "specs.py",
}
COMPONENT_KEY = "_component_"


def _is_component_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == COMPONENT_KEY


def _source_location(path: Path, node: ast.AST) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{getattr(node, 'lineno', '?')}"


def _direct_component_key_accesses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_component_key(node.slice):
            violations.append(f"{_source_location(path, node)} subscript access")
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in {"get", "pop", "setdefault"}
                and node.args
                and _is_component_key(node.args[0])
            ):
                violations.append(f"{_source_location(path, node)} {func.attr}() access")
        elif isinstance(node, ast.Compare):
            if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
                if _is_component_key(node.left) or any(
                    _is_component_key(comparator) for comparator in node.comparators
                ):
                    violations.append(f"{_source_location(path, node)} membership test")
        elif isinstance(node, ast.Dict):
            if any(_is_component_key(key) for key in node.keys if key is not None):
                violations.append(f"{_source_location(path, node)} dict literal")

    return violations


def test_runtime_modules_do_not_directly_access_component_key() -> None:
    """Runtime modules consume typed component specs, not raw YAML dict keys."""
    violations: list[str] = []
    for root in RUNTIME_ROOTS:
        paths = root.rglob("*.py") if root.is_dir() else (root,)
        for path in paths:
            if path in SERIALIZATION_HELPERS:
                continue
            violations.extend(_direct_component_key_accesses(path))

    assert violations == []
