"""Metadata extraction for coarse chunks of the kernelpack source tree."""

from __future__ import annotations

import ast
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from kernelpack_rag.chunking.coarse import CoarseChunk
from kernelpack_rag.tokenizer import tokenize


KP_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

_DATA_FILE = Path(__file__).parent.parent / "data" / "math_terms.json"
_LEXICON: list[str] = json.loads(_DATA_FILE.read_text())
_LEXICON_SORTED: list[str] = sorted(_LEXICON, key=len, reverse=True)
_NUMBA_NAMES = frozenset({"njit", "jit", "vectorize", "guvectorize"})


@dataclass
class ChunkMetadata:
    module: str
    function_name: str
    source_file: str
    line_range: tuple[int, int]
    chunk_type: str
    parent_class: str | None
    math_terms: list[str]
    cross_refs: list[str]
    cross_ref_ids: list[uuid.UUID]
    has_numba: bool


def build_symbol_table(package_root: Path) -> dict[str, uuid.UUID]:
    """Walk package_root, collect qualified names, return name→UUID mapping.

    source_file in the UUID formula is relative to the repo root
    (package_root.parent.parent), NOT an absolute path.
    Dunder methods are excluded; private functions are included.
    """
    repo_root = package_root.parent.parent
    table: dict[str, uuid.UUID] = {}

    for py_file in sorted(package_root.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue

        rel_parts = list(py_file.relative_to(package_root.parent).parts)
        rel_parts[-1] = rel_parts[-1][:-3]          # strip .py
        module = ".".join(rel_parts)
        source_file = str(py_file.relative_to(repo_root))

        try:
            tree = ast.parse(py_file.read_text())
        except SyntaxError:
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not _is_dunder(node.name):
                    qn = f"{module}.{node.name}"
                    table[qn] = uuid.uuid5(
                        KP_NAMESPACE, f"code:{source_file}:{qn}:coarse:0"
                    )

            elif isinstance(node, ast.ClassDef):
                cls_qn = f"{module}.{node.name}"
                table[cls_qn] = uuid.uuid5(
                    KP_NAMESPACE, f"code:{source_file}:{cls_qn}:coarse:0"
                )
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not _is_dunder(child.name):
                            qn = f"{module}.{node.name}.{child.name}"
                            table[qn] = uuid.uuid5(
                                KP_NAMESPACE, f"code:{source_file}:{qn}:coarse:0"
                            )

    out_path = Path(__file__).parent.parent / "data" / "symbol_table.json"
    out_path.write_text(
        json.dumps({k: str(v) for k, v in table.items()}, indent=2)
    )

    return table


def extract_metadata(chunk: CoarseChunk, symbol_table: dict) -> ChunkMetadata:
    """Extract rich metadata from a coarse chunk.

    Reads source from disk (does not use chunk.ast_node) and parses AST
    fresh from chunk.text via ast.parse to detect decorators and calls.
    """
    module = chunk.module
    qualname = chunk.qualname
    suffix = qualname[len(module) + 1:] if qualname.startswith(module + ".") else qualname
    parts = suffix.split(".")

    if len(parts) == 1:
        function_name = parts[0]
        parent_class = None
        chunk_type = "class" if (function_name and function_name[0].isupper()) else "function"
    else:
        parent_class = parts[0]
        function_name = parts[-1]
        chunk_type = "method"

    # Read source file and get chunk text from line_range.
    src_text = Path(chunk.source_file).read_text()
    lo, hi = chunk.line_range
    chunk_text = "\n".join(src_text.splitlines()[lo - 1 : hi])

    # Parse full file AST to find the node (captures decorators accurately).
    full_tree = ast.parse(src_text)
    ast_node = _find_node(full_tree, parts)

    has_numba = isinstance(
        ast_node, (ast.FunctionDef, ast.AsyncFunctionDef)
    ) and _has_numba_decorator(ast_node.decorator_list)

    cross_refs, cross_ref_ids = _extract_cross_refs(ast_node, module, symbol_table)

    return ChunkMetadata(
        module=module,
        function_name=function_name,
        source_file=chunk.source_file,
        line_range=chunk.line_range,
        chunk_type=chunk_type,
        parent_class=parent_class,
        math_terms=_extract_math_terms(chunk_text),
        cross_refs=cross_refs,
        cross_ref_ids=cross_ref_ids,
        has_numba=has_numba,
    )


# ── helpers ────────────────────────────────────────────────────────────────────


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _find_node(
    tree: ast.Module, parts: list[str]
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | None:
    """Return the AST node matching the suffix parts (after the module)."""
    if len(parts) == 1:
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and node.name == parts[0]:
                return node
    elif len(parts) >= 2:
        cls_name, method_name = parts[0], parts[-1]
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for child in ast.iter_child_nodes(node):
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) and child.name == method_name:
                        return child
    return None


def _has_numba_decorator(decorators: list) -> bool:
    for dec in decorators:
        if isinstance(dec, ast.Name) and dec.id in _NUMBA_NAMES:
            return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name) and func.id in _NUMBA_NAMES:
                return True
            if isinstance(func, ast.Attribute) and func.attr in _NUMBA_NAMES:
                return True
        if isinstance(dec, ast.Attribute) and dec.attr in _NUMBA_NAMES:
            return True
    return False


def _extract_math_terms(text: str) -> list[str]:
    """Return lexicon terms found in text (no duplicates, all lowercase)."""
    text_lower = text.lower()
    raw_tokens: set[str] = set(tokenize(text))

    found: list[str] = []
    seen: set[str] = set()
    for term in _LEXICON_SORTED:
        tl = term.lower()
        if tl in seen:
            continue
        if " " in tl or "-" in tl:
            if tl in text_lower:
                found.append(tl)
                seen.add(tl)
        else:
            if tl in raw_tokens:
                found.append(tl)
                seen.add(tl)
    return found


def _extract_cross_refs(
    node, module: str, symbol_table: dict
) -> tuple[list[str], list[uuid.UUID]]:
    """Find intra-package function/class calls in node that exist in symbol_table."""
    if node is None:
        return [], []

    found: set[str] = set()
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name: str | None = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr

        if name:
            for key in symbol_table:
                if key.split(".")[-1] == name:
                    found.add(key)

    refs = sorted(found)
    ids = [symbol_table[r] for r in refs]
    return refs, ids


def get_relative_source_file(source_file: str) -> str:
    path = Path(source_file)
    for p in [path] + list(path.parents):
        if p.name == "src":
            return str(path.relative_to(p.parent))
    return str(path)


def get_coarse_uuid(source_file: str, qualname: str, module: str) -> uuid.UUID:
    rel_path = get_relative_source_file(source_file)
    suffix = qualname[len(module) + 1:] if qualname.startswith(module + ".") else qualname
    qn = f"{module}.{suffix}"
    return uuid.uuid5(KP_NAMESPACE, f"code:{rel_path}:{qn}:coarse:0")


def get_fine_uuid(source_file: str, qualname: str, module: str, window_idx: int) -> uuid.UUID:
    rel_path = get_relative_source_file(source_file)
    suffix = qualname[len(module) + 1:] if qualname.startswith(module + ".") else qualname
    qn = f"{module}.{suffix}"
    return uuid.uuid5(KP_NAMESPACE, f"code:{rel_path}:{qn}:fine:{window_idx}")

