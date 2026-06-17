"""Fine-grained source-code chunking for coarse Python chunks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from kernelpack_rag.chunking.coarse import CoarseChunk
from kernelpack_rag.chunking.metadata import KP_NAMESPACE, get_coarse_uuid



@dataclass
class FineChunk:
    text: str
    qualname: str
    chunk_type: str
    line_range: tuple[int, int]
    source_file: str
    parent_class: str | None
    parent_id: uuid.UUID
    window_idx: int
    granularity: str = "fine"


def fine_chunks(coarse: CoarseChunk) -> list[FineChunk]:
    """Split a function or method coarse chunk into statement-aligned windows."""
    if coarse.chunk_type in {"class_header", "module_docstring"}:
        return []

    if coarse.chunk_type not in {"function", "method"}:
        return []

    lines = coarse.text.splitlines()
    if not lines:
        return []

    statements = _body_statement_ranges(coarse.text)
    if not statements:
        statements = [(1, len(lines))]

    windows = _window_ranges(statements, len(lines))
    parent_id = get_coarse_uuid(coarse.source_file, coarse.qualname, coarse.module)
    line_offset = coarse.line_range[0] - 1

    return [
        FineChunk(
            text="\n".join(lines[start - 1 : end]).rstrip("\n"),
            qualname=coarse.qualname,
            chunk_type="fine",
            line_range=(start + line_offset, end + line_offset),
            source_file=coarse.source_file,
            parent_class=coarse.parent_class,
            parent_id=parent_id,
            window_idx=idx,
        )
        for idx, (start, end) in enumerate(windows)
    ]


def _tree_sitter_parser() -> Any:
    from tree_sitter import Language, Parser
    import tree_sitter_python

    language = tree_sitter_python.language()
    if not isinstance(language, Language):
        language = Language(language)

    try:
        return Parser(language)
    except TypeError:
        parser = Parser()
        parser.set_language(language)
        return parser


def _body_statement_ranges(source: str) -> list[tuple[int, int]]:
    parser = _tree_sitter_parser()
    root = parser.parse(source.encode()).root_node
    function_node = _function_definition_node(root)
    if function_node is None:
        return []

    body = function_node.child_by_field_name("body")
    if body is None:
        return []

    return [
        (statement.start_point[0] + 1, statement.end_point[0] + 1)
        for statement in body.named_children
    ]


def _function_definition_node(root: Any) -> Any | None:
    for child in root.named_children:
        if child.type == "function_definition":
            return child
        if child.type == "decorated_definition":
            for grandchild in child.named_children:
                if grandchild.type == "function_definition":
                    return grandchild
    return None


def _window_ranges(
    statement_ranges: list[tuple[int, int]], line_count: int
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    current_start = 1
    current_end = 0
    statements_in_window = 0

    for idx, (_statement_start, statement_end) in enumerate(statement_ranges):
        # _statement_start intentionally unused — windows anchor to current_start,
        # which absorbs any blank lines between statements.
        proposed_size = statement_end - current_start + 1
        current_size = current_end - current_start + 1
        can_close = (
            statements_in_window > 0 and proposed_size > 5 and current_size >= 3
        )

        if can_close:
            windows.append((current_start, current_end))
            current_start = current_end + 1
            current_end = statement_end
            statements_in_window = 1
        else:
            current_end = statement_end
            statements_in_window += 1

        if idx == len(statement_ranges) - 1:
            current_end = line_count

    if statements_in_window > 0:
        windows.append((current_start, current_end))

    return windows
