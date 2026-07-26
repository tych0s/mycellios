#!/usr/bin/env python3
"""Extract Mycellios package imports with CPython's Python 3.12 AST."""

from __future__ import annotations

import ast
import json
import sys
from typing import Any


SCHEMA = "mycellios-native-python-import-ast/1"
PACKAGE = "distributed_runtime"
PACKAGE_PREFIX = f"{PACKAGE}."
API_FUNCTIONS = {
    "builtins": frozenset({"__import__"}),
    "importlib": frozenset({"import_module"}),
    "runpy": frozenset({"run_module"}),
}


class AnalysisError(Exception):
    """The request or Python source cannot be analyzed safely."""


class ImportAnalyzer:
    def __init__(self, tree: ast.AST) -> None:
        self._tree = tree
        self._imports: set[str] = set()
        self._module_aliases: dict[str, set[str]] = {
            module: {module} for module in API_FUNCTIONS
        }
        self._function_aliases: dict[str, set[str]] = {
            "__import__": {"builtins.__import__"}
        }
        self._collect_api_aliases()

    def analyze(self) -> list[str]:
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                self._visit_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._visit_import_from(node)
            elif isinstance(node, ast.Call):
                self._visit_call(node)
        return sorted(self._imports)

    def _collect_api_aliases(self) -> None:
        assignments: list[tuple[ast.expr, ast.expr]] = []
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in API_FUNCTIONS:
                        bound_name = alias.asname or alias.name
                        self._add_alias(
                            self._module_aliases,
                            bound_name,
                            {alias.name},
                        )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module in API_FUNCTIONS
            ):
                for alias in node.names:
                    if alias.name in API_FUNCTIONS[node.module]:
                        bound_name = alias.asname or alias.name
                        self._add_alias(
                            self._function_aliases,
                            bound_name,
                            {f"{node.module}.{alias.name}"},
                        )
            elif isinstance(node, ast.Assign):
                assignments.extend((target, node.value) for target in node.targets)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                assignments.append((node.target, node.value))
            elif isinstance(node, ast.NamedExpr):
                assignments.append((node.target, node.value))

        changed = True
        while changed:
            changed = False
            for target, value in assignments:
                changed |= self._bind_assignment(target, value)

    def _bind_assignment(self, target: ast.expr, value: ast.expr) -> bool:
        if isinstance(target, ast.Name):
            changed = self._add_alias(
                self._module_aliases,
                target.id,
                self._resolve_modules(value),
            )
            changed |= self._add_alias(
                self._function_aliases,
                target.id,
                self._resolve_functions(value),
            )
            return changed
        if (
            isinstance(target, (ast.List, ast.Tuple))
            and isinstance(value, (ast.List, ast.Tuple))
            and len(target.elts) == len(value.elts)
        ):
            changed = False
            for child_target, child_value in zip(target.elts, value.elts):
                changed |= self._bind_assignment(child_target, child_value)
            return changed
        return False

    @staticmethod
    def _add_alias(
        aliases: dict[str, set[str]],
        name: str,
        values: set[str],
    ) -> bool:
        if not values:
            return False
        existing = aliases.setdefault(name, set())
        before = len(existing)
        existing.update(values)
        return len(existing) != before

    def _resolve_modules(self, expression: ast.expr) -> set[str]:
        if isinstance(expression, ast.Name):
            return set(self._module_aliases.get(expression.id, ()))
        return set()

    def _resolve_functions(self, expression: ast.expr) -> set[str]:
        if isinstance(expression, ast.Name):
            return set(self._function_aliases.get(expression.id, ()))
        if isinstance(expression, ast.Attribute):
            resolved: set[str] = set()
            for module in self._resolve_modules(expression.value):
                if expression.attr in API_FUNCTIONS.get(module, ()):
                    resolved.add(f"{module}.{expression.attr}")
            return resolved
        return set()

    def _visit_import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._record_absolute_module(alias.name)

    def _visit_import_from(self, node: ast.ImportFrom) -> None:
        if node.level > 0:
            if node.level != 1:
                raise AnalysisError(
                    "relative import escapes the distributed_runtime package"
                )
            if node.module:
                self._record_absolute_module(f"{PACKAGE}.{node.module}")
            else:
                for alias in node.names:
                    if alias.name == "*":
                        self._imports.add(f"{PACKAGE}.__init__")
                    else:
                        self._record_absolute_module(
                            f"{PACKAGE}.{alias.name}"
                        )
            return

        module = node.module or ""
        if module == PACKAGE:
            for alias in node.names:
                if alias.name == "*":
                    self._imports.add(f"{PACKAGE}.__init__")
                else:
                    self._record_absolute_module(
                        f"{PACKAGE}.{alias.name}"
                    )
        elif module.startswith(PACKAGE_PREFIX):
            self._record_absolute_module(module)

    def _visit_call(self, node: ast.Call) -> None:
        for function in self._resolve_functions(node.func):
            argument_name = {
                "builtins.__import__": "name",
                "importlib.import_module": "name",
                "runpy.run_module": "mod_name",
            }[function]
            module_name = self._call_string_argument(node, argument_name)
            if module_name is None:
                raise AnalysisError(
                    f"{function} target must be a statically known string"
                )
            if module_name.startswith("."):
                if function != "importlib.import_module":
                    continue
                package = self._call_string_argument(node, "package", 1)
                if package is None:
                    package_node = self._call_argument(node, "package", 1)
                    if isinstance(package_node, ast.Name) and package_node.id == "__package__":
                        package = PACKAGE
                if (
                    package != PACKAGE
                    or not module_name.startswith(".")
                    or module_name.startswith("..")
                ):
                    raise AnalysisError(
                        "cannot resolve relative dynamic import inside distributed_runtime"
                    )
                module_name = f"{PACKAGE}.{module_name[1:]}"
            self._record_absolute_module(module_name)

    @staticmethod
    def _call_argument(
        node: ast.Call,
        keyword_name: str,
        position: int = 0,
    ) -> ast.expr | None:
        if len(node.args) > position:
            return node.args[position]
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                return keyword.value
        return None

    def _call_string_argument(
        self,
        node: ast.Call,
        keyword_name: str,
        position: int = 0,
    ) -> str | None:
        return self._static_string(
            self._call_argument(node, keyword_name, position)
        )

    def _static_string(self, expression: ast.expr | None) -> str | None:
        if (
            isinstance(expression, ast.Constant)
            and isinstance(expression.value, str)
        ):
            return expression.value
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            left = self._static_string(expression.left)
            right = self._static_string(expression.right)
            if left is not None and right is not None:
                return left + right
        return None

    def _record_absolute_module(self, module_name: str) -> None:
        if module_name == PACKAGE:
            self._imports.add(f"{PACKAGE}.__init__")
            return
        if not module_name.startswith(PACKAGE_PREFIX):
            return
        segments = module_name.split(".")
        if any(not segment.isidentifier() for segment in segments):
            raise AnalysisError(
                f"invalid distributed_runtime module name: {module_name!r}"
            )
        self._imports.add(module_name)


def _read_request() -> list[dict[str, str]]:
    try:
        request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"invalid analyzer request: {error}") from error
    if (
        not isinstance(request, dict)
        or set(request) != {"schema", "sources"}
        or request.get("schema") != SCHEMA
        or not isinstance(request.get("sources"), list)
        or not request["sources"]
    ):
        raise AnalysisError("invalid analyzer request structure")

    sources: list[dict[str, str]] = []
    observed_paths: set[str] = set()
    for entry in request["sources"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "source"}
            or not isinstance(entry.get("path"), str)
            or not entry["path"]
            or len(entry["path"]) > 4096
            or any(ord(character) < 32 or ord(character) == 127 for character in entry["path"])
            or entry["path"] in observed_paths
            or not isinstance(entry.get("source"), str)
        ):
            raise AnalysisError("invalid analyzer source entry")
        observed_paths.add(entry["path"])
        sources.append(entry)
    return sources


def _analyze_source(path: str, source: str) -> dict[str, Any]:
    try:
        tree = ast.parse(
            source,
            filename=path,
            mode="exec",
            feature_version=(3, 12),
        )
    except SyntaxError as error:
        location = f"{path}:{error.lineno or 0}:{error.offset or 0}"
        raise AnalysisError(f"{location}: {error.msg}") from error
    return {
        "path": path,
        "imports": ImportAnalyzer(tree).analyze(),
    }


def main() -> int:
    if sys.version_info[:2] < (3, 12) or sys.version_info.major != 3:
        raise AnalysisError("Python 3.12 or newer is required")
    sources = _read_request()
    response = {
        "schema": SCHEMA,
        "pythonVersion": [sys.version_info.major, sys.version_info.minor],
        "sources": [
            _analyze_source(entry["path"], entry["source"])
            for entry in sources
        ],
    }
    encoded = json.dumps(
        response,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    sys.stdout.buffer.write(encoded + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as error:
        sys.stderr.write(f"native Python AST analysis failed: {error}\n")
        raise SystemExit(2) from None
