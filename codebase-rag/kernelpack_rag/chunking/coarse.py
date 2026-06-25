"""Coarse source-code chunking for Python files."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Kept for backward-compat (notebook imports this); no longer used as a gate.
MIN_LINES = 5


@dataclass
class CoarseChunk:
    text: str
    qualname: str
    chunk_type: str
    line_range: tuple[int, int]
    source_file: str
    parent_class: str | None
    module: str

def _source_file_to_module(path: Path) -> str:
    parts = path.with_suffix("").parts
    idx = next((i for i, p in enumerate(parts) if p == "kernelpack"), None)
    if idx is None:
        return path.stem
    module_parts = list(parts[idx:])
    if module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]
    return ".".join(module_parts) if module_parts else "kernelpack"

def chunk_file(path: Path) -> list[CoarseChunk]:
    source = path.read_text()
    if path.name == "__init__.py":
        return _chunk_init_file(path, source)
    parser = _tree_sitter_parser()
    if parser is None:
        raise RuntimeError(
            "tree-sitter or tree-sitter-python is not installed. "
            "Run: pip install tree-sitter tree-sitter-python"
        )
    return _chunk_file_with_tree_sitter(path, source, parser)


def _chunk_init_file(path: Path, source: str) -> list[CoarseChunk]:
    """Return a single module-exports chunk for an __init__.py file."""
    if not source.strip():
        return []
    module = _source_file_to_module(path)
    lines = source.splitlines()
    return [
        CoarseChunk(
            text=source.rstrip("\n"),
            qualname=f"{module}.__init__",
            chunk_type="module_exports",
            line_range=(1, len(lines)),
            source_file=str(path),
            parent_class=None,
            module=module,
        )
    ]


def _tree_sitter_parser() -> Any | None:
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python
    except ImportError:
        return None

    language = tree_sitter_python.language()
    if not isinstance(language, Language):
        language = Language(language)

    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.set_language(language)
        return parser


def _chunk_file_with_tree_sitter(
    path: Path, source: str, parser: Any
) -> list[CoarseChunk]:
    source_bytes = source.encode()
    root = parser.parse(source_bytes).root_node
    chunks: list[CoarseChunk] = []

    docstring = _module_docstring_node(root)
    if docstring is not None:
        chunks.append(
            _make_chunk(
                path=path,
                source=source,
                node=docstring,
                qualname=_source_file_to_module(path),
                chunk_type="module_docstring",
                parent_class=None,
            )
        )

    for outer_node, definition_node in _definition_children(root):
        if definition_node.type == "function_definition":
            chunks.append(
                _make_chunk(
                    path=path,
                    source=source,
                    node=outer_node,
                    qualname=f"{_node_name(definition_node)}",
                    chunk_type="function",
                    parent_class=None,
                )
            )
        elif definition_node.type == "class_definition":
            chunks.extend(_class_chunks(path, source, outer_node, definition_node))

    return chunks


def _class_chunks(
    path: Path, source: str, outer_node: Any, class_node: Any
) -> list[CoarseChunk]:
    class_name = _node_name(class_node)
    method_chunks: list[CoarseChunk] = []
    long_method_ranges: list[tuple[int, int]] = []

    body = class_node.child_by_field_name("body")
    if body is not None:
        for method_outer, method_node in _definition_children(body):
            if method_node.type != "function_definition":
                continue

            long_method_ranges.append(
                (method_outer.start_point[0] + 1, method_outer.end_point[0] + 1)
            )
            method_name = _node_name(method_node)
            method_chunks.append(
                _make_chunk(
                    path=path,
                    source=source,
                    node=method_outer,
                    qualname=f"{class_name}.{method_name}",
                    chunk_type="method",
                    parent_class=class_name,
                )
            )

    class_chunk = CoarseChunk(
        text=_slice_without_ranges(source, outer_node, long_method_ranges),
        qualname=f"{class_name}",
        chunk_type="class_header",
        line_range=(outer_node.start_point[0] + 1, outer_node.end_point[0] + 1),
        source_file=str(path),
        parent_class=None,
        module=_source_file_to_module(path),
    )
    return [class_chunk, *method_chunks]


def _definition_children(node: Any) -> list[tuple[Any, Any]]:
    definitions: list[tuple[Any, Any]] = []
    for child in node.named_children:
        if child.type in {"class_definition", "function_definition"}:
            definitions.append((child, child))
        elif child.type == "decorated_definition":
            definition = next(
                (
                    grandchild
                    for grandchild in child.named_children
                    if grandchild.type in {"class_definition", "function_definition"}
                ),
                None,
            )
            if definition is not None:
                definitions.append((child, definition))
    return definitions


def _module_docstring_node(root: Any) -> Any | None:
    if not root.named_children:
        return None

    first = root.named_children[0]
    if first.type != "expression_statement":
        return None
    if any(child.type == "string" for child in first.named_children):
        return first
    return None


def _node_name(node: Any) -> str:
    name = node.child_by_field_name("name")
    if name is None:
        return ""
    return name.text.decode()


def _make_chunk(
    *,
    path: Path,
    source: str,
    node: Any,
    qualname: str,
    chunk_type: str,
    parent_class: str | None,
) -> CoarseChunk:
    return CoarseChunk(
        text=_slice_node(source, node),
        qualname=qualname,
        chunk_type=chunk_type,
        line_range=(node.start_point[0] + 1, node.end_point[0] + 1),
        source_file=str(path),
        parent_class=parent_class,
        module=_source_file_to_module(path),
    )


def _line_count(node: Any) -> int:
    return node.end_point[0] - node.start_point[0] + 1


def _method_line_count(source: str, method_node: Any, class_node: Any) -> int:
    line_count = _line_count(method_node)
    lines = source.splitlines()
    next_line_index = method_node.end_point[0] + 1
    if (
        next_line_index < class_node.end_point[0]
        and next_line_index < len(lines)
        and not lines[next_line_index].strip()
    ):
        return line_count + 1
    return line_count


def _slice_node(source: str, node: Any) -> str:
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.start_point[0] : node.end_point[0] + 1]).rstrip("\n")


def _slice_without_ranges(
    source: str, node: Any, excluded_ranges: list[tuple[int, int]]
) -> str:
    excluded = {
        line_number
        for start, end in excluded_ranges
        for line_number in range(start, end + 1)
    }
    kept = [
        line
        for line_number, line in enumerate(source.splitlines(keepends=True), start=1)
        if node.start_point[0] + 1 <= line_number <= node.end_point[0] + 1
        and line_number not in excluded
    ]
    return "".join(kept).rstrip("\n")

