"""Identifier-aware tokenizer for sparse BM25 code-search vectors.

WHY THIS EXISTS
---------------
Standard whitespace tokenization kills BM25 signal on code identifiers.

With str.split(), indexing a chunk containing 'VariablePoissonSolver.solve'
produces one token: 'VariablePoissonSolver.solve'. A query written as
'VariablePoissonSolver solve' (space-separated) produces 'VariablePoissonSolver'
with no dot. The two strings never match. BM25 scores every chunk near-zero,
the sparse retrieval leg is functionally dead, and hybrid search silently
degrades to dense-only with RRF overhead.

This tokenizer fixes that by emitting identifiers at multiple granularities
so that index and query always share overlapping tokens regardless of how the
caller wrote the identifier.

EMISSION STRATEGY (applied to each source lexical token)
---------------------------------------------------------
For each token from lexical_split (dots preserved, punctuation stripped):

    1. Whole lowercased token        'variablepoissonsolver.solve'  — exact lookup
    2. Each dot-segment lowercased   'variablepoissonsolver', 'solve' — class/method lookup
    3. Snake/camel fragments         'variable', 'poisson', 'solver'  — NL query lookup

Duplicates within a single source token are suppressed so TF reflects actual
word frequency in the text, not emission multiplicity. The same string can
still accumulate across multiple source tokens (correct BM25 behavior).

Example:
    tokenize("PoissonSolver.solve")
    → {"poissonsolver.solve", "poissonsolver", "poisson", "solver", "solve"}

    tokenize("alpha alpha")         # "alpha" from two source tokens
    → Counter({"alpha": 2})         # TF=2, not 4 or 6

SPARSE VECTOR FORMAT
--------------------
build_sparse_vector returns {stable_u32_hash(token): raw_term_frequency}.
IDF is NOT applied here — Qdrant applies it server-side via modifier: idf
on the bm25_code sparse space. Never compute IDF client-side.

NOTE: build_sparse_vector is here temporarily. It moves to embed/sparse.py
at Step 11 of the foundation plan.

MODULE BOUNDARIES
-----------------
This module is pure functions with no external dependencies. Do not import
Qdrant, model libraries, or anything project-specific here. Two downstream
consumers:

    embed/sparse.py (Step 11)       — builds Qdrant sparse vectors from this output
    chunking/metadata.py (Step 7)   — matches tokenizer fragments against the
                                      domain math-terms lexicon (rbf, laplacian, etc.)

TESTS
-----
tests/test_tokenizer.py — run with: pytest tests/test_tokenizer.py -v
The BM25BugRegression class documents the historical failure mode. Do not
delete or weaken those tests.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter


_LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
_CAMEL_BOUNDARY_RE = re.compile(
    r"""
    [A-Z]+(?=[A-Z][a-z]|[0-9_]|$) |
    [A-Z]?[a-z]+ |
    [0-9]+
    """,
    re.VERBOSE,
)


def lexical_split(text: str):
    """Split on whitespace and punctuation while preserving dotted names."""
    for match in _LEXICAL_TOKEN_RE.finditer(text):
        yield match.group(0)


def split_dots(token: str):
    """Split a token on dots, omitting empty parts."""
    for part in token.split("."):
        if part:
            yield part


def split_snake_camel(token: str):
    """Split snake_case and CamelCase-ish identifiers into lowercase pieces."""
    for snake_part in token.split("_"):
        if not snake_part:
            continue
        for match in _CAMEL_BOUNDARY_RE.finditer(snake_part):
            part = match.group(0).lower()
            if part:
                yield part


def tokenize(text: str):
    for tok in lexical_split(text):
        seen: set[str] = set()
        candidates = [tok.lower()]
        for part in split_dots(tok):
            candidates.append(part.lower())
            candidates.extend(split_snake_camel(part))
        for t in candidates:
            if t not in seen:
                seen.add(t)
                yield t


def bm25_tokenize(text: str) -> list[str]:
    """Return the flat token list used for BM25-style term counting."""
    return list(tokenize(text))


def stable_u32_hash(token: str) -> int:
    """Return a deterministic unsigned 32-bit hash for a token."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little", signed=False)


def build_sparse_vector(text: str) -> dict[int, float]:
    """Return {stable_u32_hash(token): raw_term_frequency} for sparse search."""
    counts = Counter(bm25_tokenize(text))
    return {stable_u32_hash(token): float(count) for token, count in counts.items()}