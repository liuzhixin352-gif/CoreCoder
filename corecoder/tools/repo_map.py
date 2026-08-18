"""Repository structure mapping tool."""

import ast
import json
import os
from pathlib import Path

from corecoder.permissions import ToolPermission

from .base import Tool


_SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "agent_env",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
}


def _collect_python_files(root: Path, max_files: int) -> tuple[list[Path], bool]:
    """Collect Python files while skipping generated and dependency directories."""
    if root.is_file():
        if root.suffix == ".py":
            return [root], False
        return [], False

    python_files: list[Path] = []

    for current_dir, dir_names, file_names in os.walk(root):
        dir_names[:] = sorted(
            directory
            for directory in dir_names
            if directory not in _SKIP_DIRS
        )

        for file_name in sorted(file_names):
            if not file_name.endswith(".py"):
                continue

            python_files.append(Path(current_dir) / file_name)

            if len(python_files) > max_files:
                return python_files[:max_files], True

    return python_files, False


def _extract_imports(tree: ast.Module) -> list[str]:
    """Extract top-level imported module names."""
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in imports:
                    imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module_name = "." * node.level + (node.module or "")

            if module_name and module_name not in imports:
                imports.append(module_name)

    return imports


def _extract_file_info(
    file_path: Path,
    display_root: Path,
    tree: ast.Module,
) -> dict:
    """Extract top-level symbols from one parsed Python file."""
    classes: list[str] = []
    functions: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)

    return {
        "path": file_path.relative_to(display_root).as_posix(),
        "classes": classes,
        "functions": functions,
        "imports": _extract_imports(tree),
    }


class RepoMapTool(Tool):
    """Build a structured map of Python files in a repository."""

    name = "repo_map"
    permission = ToolPermission.READ

    description = (
        "Scan a repository and return a structured map of Python files, "
        "including top-level classes, functions, and imports. "
        "Use this to understand repository structure before editing code."
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository directory or Python file to scan (default: current directory)",
            },
            "max_files": {
                "type": "integer",
                "description": "Maximum number of Python files to scan (default: 500)",
            },
        },
        "required": [],
    }

    def execute(self, path: str = ".", max_files: int = 500) -> str:
        """Scan Python files and return repository information as JSON text."""
        root = Path(path).expanduser().resolve()

        if not root.exists():
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Path not found: {path}",
                },
                ensure_ascii=False,
                indent=2,
            )

        if max_files < 1:
            return json.dumps(
                {
                    "status": "error",
                    "error": "max_files must be at least 1",
                },
                ensure_ascii=False,
                indent=2,
            )

        if root.is_file() and root.suffix != ".py":
            return json.dumps(
                {
                    "status": "error",
                    "error": f"Not a Python file: {path}",
                },
                ensure_ascii=False,
                indent=2,
            )

        display_root = root if root.is_dir() else root.parent
        python_files, truncated = _collect_python_files(root, max_files)

        files: list[dict] = []
        syntax_errors: list[dict] = []
        read_errors: list[dict] = []

        for file_path in python_files:
            relative_path = file_path.relative_to(display_root).as_posix()

            try:
                source = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                read_errors.append(
                    {
                        "path": relative_path,
                        "error": str(exc),
                    }
                )
                continue

            try:
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError as exc:
                syntax_errors.append(
                    {
                        "path": relative_path,
                        "line": exc.lineno,
                        "message": exc.msg,
                    }
                )
                continue

            files.append(
                _extract_file_info(
                    file_path=file_path,
                    display_root=display_root,
                    tree=tree,
                )
            )

        result = {
            "status": "ok",
            "root": str(root),
            "file_count": len(python_files),
            "parsed_file_count": len(files),
            "truncated": truncated,
            "max_files": max_files,
            "files": files,
            "syntax_errors": syntax_errors,
            "read_errors": read_errors,
        }

        return json.dumps(result, ensure_ascii=False, indent=2)