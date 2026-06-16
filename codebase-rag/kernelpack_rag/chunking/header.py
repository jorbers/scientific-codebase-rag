"""Context header builder for coarse chunks."""

from __future__ import annotations

import ast
import re

from kernelpack_rag.chunking.coarse import CoarseChunk

def build_header(chunk: CoarseChunk, source: str, all_chunks: list[CoarseChunk]) -> str:
    """Build the optional context header for a coarse chunk."""
    lines: list[str] = []

    if chunk.parent_class is not None:
        lines.append(f"# class: {chunk.parent_class}")

    for import_line, usable_names in _import_lines(source):
        if any(_name_referenced(name, chunk.text) for name in usable_names):
            lines.append(f"# {import_line}")

    above, below = _neighbors(chunk, all_chunks)
    if above is not None:
        signature = _signature(above)
        if signature is not None:
            lines.append(f"# above: {signature}")
    if below is not None:
        signature = _signature(below)
        if signature is not None:
            lines.append(f"# below: {signature}")

    return "\n".join(lines)


def _import_lines(source: str) -> list[tuple[str, list[str]]]:
    imports: list[tuple[str, list[str]]] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            usable_names = _usable_names_from_node(node)
            if usable_names:
                imports.append((ast.unparse(node), usable_names))

    return imports


def _usable_names_from_node(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [
            alias.asname if alias.asname is not None else alias.name.split(".", 1)[0]
            for alias in node.names
        ]
    if isinstance(node, ast.ImportFrom):
        return [
            alias.asname if alias.asname is not None else alias.name
            for alias in node.names
            if alias.name != "*"
        ]
    return []


def _neighbors(
    chunk: CoarseChunk, all_chunks: list[CoarseChunk]
) -> tuple[CoarseChunk | None, CoarseChunk | None]:
    candidates = sorted(
        (
            candidate
            for candidate in all_chunks
            if candidate.chunk_type in {"function", "method"}
        ),
        key=lambda candidate: candidate.line_range[0],
    )
    chunk_start = chunk.line_range[0]

    above = next(
        (
            candidate
            for candidate in reversed(candidates)
            if candidate.line_range[0] < chunk_start
        ),
        None,
    )
    below = next(
        (
            candidate
            for candidate in candidates
            if candidate.line_range[0] > chunk_start
        ),
        None,
    )
    return above, below


def _signature(chunk: CoarseChunk) -> str | None:
    for line in chunk.text.splitlines():
        stripped = line.lstrip()
        if re.match(r"(?:def|async\s+def)\b", stripped):
            return stripped
    return None


def _name_referenced(name: str, text: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None
