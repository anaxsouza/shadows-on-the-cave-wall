"""
Architecture test: verify no file imports from non-existent directories.

These directories are documented in AGENTS.md as non-existent.
Imports from them are runtime bugs (C1, C3).
"""

import ast
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"

NONEXISTENT_DIRS = [
    "src.shared.logging",
    "src.infrastructure.cache",
    "src.core.model",
    "src.utils.analysis",
    "src.utils.data",
]

ALIAS_PATTERNS = {
    "src.shared.logging": [".shared.logging"],
    "src.infrastructure.cache": [".infrastructure.cache"],
    "src.core.model": [".core.model"],
    "src.utils.analysis": [".utils.analysis"],
    "src.utils.data": [".utils.data"],
}


def find_python_files(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f


def extract_imports(filepath):
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, "absolute"))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.module, "from"))
    return imports


def find_violations():
    violations = []

    for py_file in find_python_files(SRC_DIR):
        rel = str(py_file.relative_to(SRC_DIR).with_suffix("")).replace(os.sep, ".")
        short_path = str(py_file.relative_to(REPO_ROOT))

        for module_name, imp_type in extract_imports(py_file):
            for bad_dir in NONEXISTENT_DIRS:
                if module_name == bad_dir:
                    violations.append(
                        f"{short_path}: imports '{module_name}' (absolute) — directory does not exist"
                    )
                if bad_dir in ALIAS_PATTERNS:
                    for alias in ALIAS_PATTERNS[bad_dir]:
                        if module_name.endswith(alias) and imp_type == "from":
                            violations.append(
                                f"{short_path}: imports from '{module_name}' — mirrors non-existent '{bad_dir}'"
                            )
    return violations


def test_no_imports_from_nonexistent_dirs():
    violations = find_violations()
    assert len(violations) == 0, (
        f"Found {len(violations)} imports from non-existent directories:\n"
        + "\n".join(violations)
    )
