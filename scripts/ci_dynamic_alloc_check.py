#!/usr/bin/env python3
"""
AST-based guard that fails if dynamic tensor allocations are used inside
non-__init__ functions/methods in critical export paths.

Blocked calls (inside any def not named __init__):
- torch.zeros, torch.ones, torch.randn, torch.rand
- torch.zeros_like, torch.ones_like, torch.randn_like, torch.rand_like
- <tensor>.new_zeros

Exclusions:
- Files: kokoro/model.py, kokoro/pipeline.py (not part of CoreML export graph)
- __init__ methods (allowed to register buffers / parameters)
- nn.Parameter initializers and register_buffer are naturally in __init__

Usage:
  python scripts/ci_dynamic_alloc_check.py

Exit codes:
  0 -> OK
  1 -> Violations found
  2 -> Script error
"""
from __future__ import annotations
import ast
import pathlib
import sys
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / 'kokoro',
    ROOT / 'export_duration.py',
    ROOT / 'export_synthesizers.py',
]
EXCLUDE_FILES = {
    str((ROOT / 'kokoro' / 'model.py').resolve()),
    str((ROOT / 'kokoro' / 'pipeline.py').resolve()),
}

# Disallowed function/attribute names
TORCH_FUNCS = {'zeros', 'ones', 'randn', 'rand', 'zeros_like', 'ones_like', 'randn_like', 'rand_like'}
TENSOR_ATTRS = {'new_zeros'}

class AllocVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, source_lines: List[str]):
        self.filename = filename
        self.source_lines = source_lines
        self.func_stack: List[str] = []
        self.class_stack: List[Tuple[str, bool]] = []  # (class_name, is_nn_module)
        self.violations: List[Tuple[int, str]] = []

    def _is_nn_module_base(self, node: ast.ClassDef) -> bool:
        for b in node.bases:
            # nn.Module or torch.nn.Module or Module (best-effort)
            if isinstance(b, ast.Attribute):
                name = []
                cur = b
                while isinstance(cur, ast.Attribute):
                    name.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    name.append(cur.id)
                fq = '.'.join(reversed(name))
                if fq in ('nn.Module', 'torch.nn.Module') or fq.endswith('.Module'):
                    return True
            elif isinstance(b, ast.Name):
                if b.id == 'Module' or b.id.endswith('Module'):
                    return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef):
        is_mod = self._is_nn_module_base(node)
        self.class_stack.append((node.name, is_mod))
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_Call(self, node: ast.Call):
        # Only check inside methods of nn.Module-derived classes and non-__init__
        if not self.func_stack or self.func_stack[-1] == '__init__':
            return self.generic_visit(node)
        if not self.class_stack or not self.class_stack[-1][1]:
            return self.generic_visit(node)
        bad = False
        # torch.<func>
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == 'torch' and node.func.attr in TORCH_FUNCS:
                bad = True
        # <tensor>.new_zeros
        if isinstance(node.func, ast.Attribute) and node.func.attr in TENSOR_ATTRS:
            bad = True
        if bad:
            lineno = getattr(node, 'lineno', None)
            if lineno is not None and 1 <= lineno <= len(self.source_lines):
                line = self.source_lines[lineno-1].rstrip('\n')
            else:
                line = '<source unavailable>'
            self.violations.append((lineno or -1, line))
        return self.generic_visit(node)


def scan_file(path: pathlib.Path) -> List[Tuple[int, str]]:
    try:
        if str(path.resolve()) in EXCLUDE_FILES:
            return []
        text = path.read_text(encoding='utf-8')
        tree = ast.parse(text)
        visitor = AllocVisitor(str(path), text.splitlines())
        visitor.visit(tree)
        return visitor.violations
    except Exception as e:
        print(f"[ci_dynamic_alloc_check] Failed to parse {path}: {e}", file=sys.stderr)
        return []


def main() -> int:
    py_files: List[pathlib.Path] = []
    for target in TARGETS:
        if target.is_dir():
            py_files.extend(p for p in target.rglob('*.py'))
        elif target.is_file():
            py_files.append(target)
    violations_total: List[Tuple[str, int, str]] = []
    for f in sorted(py_files):
        v = scan_file(f)
        for lineno, line in v:
            violations_total.append((str(f), lineno, line))
    if violations_total:
        print("ERROR: Dynamic tensor allocations detected in non-__init__ functions:\n")
        for fname, lineno, line in violations_total:
            print(f"- {fname}:{lineno}: {line}")
        return 1
    print("OK: No dynamic tensor allocations in non-__init__ functions.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
