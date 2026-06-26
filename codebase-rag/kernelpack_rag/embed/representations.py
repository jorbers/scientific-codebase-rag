"""
Builds text representations (ctx / code / codecom / com) from chunk payloads.

Call build_representation(chunk, key) once per (chunk, named-space) pair. Returning
None means "no vector for this space" — used by com when a chunk has no comments.
"""

import ast
import re
from enum import Enum
from typing import Optional


class RepresentationKey(str, Enum):
    CTX = "ctx"
    CODE = "code"
    CODECOM = "codecom"
    COM = "com"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_representation(
    chunk: dict,
    key: "RepresentationKey | str",
) -> Optional[str]:
    """
    Build a text representation from a chunk payload dict.

    Parameters
    ----------
    chunk : dict
        Payload dict with at minimum: text, llm_summary, context_header.
        None values are treated as empty string for summary and header.
    key : RepresentationKey
        Which variant to build.

    Returns
    -------
    str | None
        The text to embed. None means "no vector for this space on this point."
        Only COM currently returns None (when there are no comments or docstrings).

    Raises
    ------
    ValueError
        If key is not a valid RepresentationKey.
    """
    if not isinstance(key, RepresentationKey):
        try:
            key = RepresentationKey(key)
        except ValueError:
            raise ValueError(
                f"'{key}' is not a valid RepresentationKey. "
                f"Valid values: {[k.value for k in RepresentationKey]}"
            )

    text: str = chunk.get("text") or ""
    llm_summary: str = chunk.get("llm_summary") or ""
    context_header: str = chunk.get("context_header") or ""

    if key is RepresentationKey.CTX:
        return _build_ctx(llm_summary, context_header, text)
    elif key is RepresentationKey.CODE:
        return _build_code(text)
    elif key is RepresentationKey.CODECOM:
        return _build_codecom(text)
    elif key is RepresentationKey.COM:
        return _build_com(text)
    else:
        raise NotImplementedError(f"RepresentationKey.{key} has no builder.")


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_ctx(summary: str, header: str, text: str) -> str:
    """
    ctx = summary + context_header + raw text.

    Ordering is from D1: summary first so truncation hits the body last.
    Parts are joined with a blank line so a code model sees them as separate blocks.
    Empty parts are dropped (no leading blank lines when summary is missing).
    """
    parts = [p for p in [summary, header, text] if p and p.strip()]
    return "\n\n".join(parts)


def _build_code(text: str) -> str:
    """
    Strip docstrings and inline comments from Python source text.

    Returns the stripped text. If stripping leaves only whitespace (degenerate chunk),
    returns the original text unchanged — better to embed something than nothing.
    """
    stripped = _strip_docstrings(text)
    stripped = _strip_inline_comments(stripped)
    stripped = stripped.strip()
    return stripped if stripped else text


def _build_codecom(text: str) -> str:
    """
    Return verbatim text.

    Today this is identical to the raw text payload. When the team adds
    annotated source (richer inline comments), this representation diverges
    from CODE automatically with no code change here.
    """
    return text


def _build_com(text: str) -> Optional[str]:
    """
    Extract docstrings and inline comments only.

    Returns None when the chunk has neither — the ingestor skips the vector.
    """
    parts: list[str] = []

    docstring = _extract_docstring(text)
    if docstring:
        parts.append(docstring)

    inline = _extract_inline_comments(text)
    if inline:
        parts.append(inline)

    if not parts:
        return None

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _extract_docstring(text: str) -> Optional[str]:
    """
    Return the first docstring from a Python source snippet, or None.

    Uses ast.get_docstring on the parsed module or first function/class body.
    Falls back gracefully if the snippet is not valid Python (e.g. a fine chunk
    that starts mid-function — ast.parse will fail on it).
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Fine chunks or header-only snippets may not be valid standalone Python.
        # Fall back to regex for docstring detection.
        return _extract_docstring_regex(text)

    # Module-level docstring
    doc = ast.get_docstring(tree)
    if doc:
        return doc

    # Function or class body docstring
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                return doc

    return None


def _extract_docstring_regex(text: str) -> Optional[str]:
    """
    Regex fallback for docstring extraction when ast.parse fails.
    Matches triple-quoted strings at the start of a function body.
    """
    pattern = r'(?:\'\'\'|""")(.*?)(?:\'\'\'|""")'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _strip_docstrings(text: str) -> str:
    """Remove docstrings from Python source text via AST line ranges; regex fallback on parse error.

    We use line-filter removal rather than ast.unparse because unparse normalizes
    whitespace and drops comments — code-pretrained models rely on indentation structure.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _strip_docstrings_regex(text)

    lines = text.splitlines(keepends=True)
    # Collect line ranges to remove (1-indexed, inclusive)
    ranges_to_remove: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not (node.body and isinstance(node.body[0], ast.Expr)):
            continue
        expr = node.body[0]
        if not isinstance(expr.value, ast.Constant) or not isinstance(expr.value.value, str):
            continue
        # expr.lineno and expr.end_lineno are 1-indexed
        ranges_to_remove.append((expr.lineno, expr.end_lineno))  # type: ignore[attr-defined]

    if not ranges_to_remove:
        return text

    # Remove lines in ranges (merge overlapping ranges to be safe)
    remove_set: set[int] = set()
    for start, end in ranges_to_remove:
        remove_set.update(range(start, end + 1))

    kept = [line for i, line in enumerate(lines, start=1) if i not in remove_set]
    return "".join(kept)


def _strip_docstrings_regex(text: str) -> str:
    """Regex fallback: remove triple-quoted strings."""
    pattern = r'(?:\'\'\'|""")(.*?)(?:\'\'\'|""")'
    return re.sub(pattern, "", text, flags=re.DOTALL)


def _strip_inline_comments(text: str) -> str:
    """
    Remove inline #-comments from each line of Python source.

    Does not remove # inside string literals (handles the common cases;
    not a full lexer). Blank lines left behind are preserved — the model
    uses indentation structure, so don't collapse the layout.
    """
    result_lines = []
    for line in text.splitlines(keepends=True):
        stripped = _strip_comment_from_line(line)
        result_lines.append(stripped)
    return "".join(result_lines)


def _strip_comment_from_line(line: str) -> str:
    """
    Strip the #-comment from a single line, preserving the newline.

    Handles the common case: `code  # comment\n` -> `code\n`
    A line that is only a comment becomes an empty line (preserves structure).
    Does not attempt to parse string literals — good enough for our chunks.
    """
    # Preserve leading whitespace for indentation
    newline = "\n" if line.endswith("\n") else ""
    content = line.rstrip("\n")

    # Find # not inside a string. Simple heuristic: track quote state.
    in_single = False
    in_double = False
    for i, ch in enumerate(content):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            # Strip comment and trailing whitespace
            return content[:i].rstrip() + newline

    return content + newline


def _extract_inline_comments(text: str) -> Optional[str]:
    """
    Extract only the inline comment text (without the # prefix) from each line.

    Returns None if no comments found.
    """
    comments = []
    for line in text.splitlines():
        in_single = False
        in_double = False
        content = line.rstrip()
        for i, ch in enumerate(content):
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif ch == "#" and not in_single and not in_double:
                comment_text = content[i + 1:].strip()
                if comment_text:
                    comments.append(comment_text)
                break

    return "\n".join(comments) if comments else None