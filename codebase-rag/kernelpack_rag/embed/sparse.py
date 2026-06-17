"""Sparse BM25 vector builder for Qdrant upsert and query use.

WHAT THIS MODULE DOES
---------------------
Converts raw chunk text into sparse vectors suitable for Qdrant's native
sparse+dense hybrid search. Two public functions:

    build_sparse_vector(text) -> dict[int, float]
        Canonical sparse representation. Maps stable_u32_hash(token) ->
        raw term frequency for every token produced by the identifier-aware
        tokenizer. No IDF — Qdrant applies it server-side via modifier: idf
        on the bm25_code and bm25_paper sparse spaces.

    to_qdrant_sparse(text) -> SparseVector
        Format adapter. Converts build_sparse_vector output into the
        qdrant_client SparseVector shape (sorted indices, parallel values)
        required for PointStruct upserts and sparse query prefetches.
        No new logic — same TF data, different shape.

PROVENANCE
----------
build_sparse_vector was implemented in tokenizer.py as a temporary home
(see that module's NOTE). This is the canonical location per the Step 11
spec. The tokenizer.py copy remains for now but this version is authoritative.
tests/test_sparse.py asserts identical output from both so they cannot silently
diverge while the tokenizer.py copy still exists.

MODULE BOUNDARIES
-----------------
Imports tokenizer primitives (bm25_tokenize, stable_u32_hash) and one
qdrant_client type (SparseVector). No model libraries, no Qdrant client
instance, no config, no I/O. The ingest orchestrator (ingest.py, Step 12)
calls to_qdrant_sparse and wires the result into PointStruct; this module
has no knowledge of collections, spaces, or point IDs.
"""

from __future__ import annotations

from collections import Counter

from qdrant_client.models import SparseVector

from kernelpack_rag.tokenizer import bm25_tokenize, stable_u32_hash



def build_sparse_vector(text: str) -> dict[int, float]:
    """Return {stable_u32_hash(token): raw_term_frequency} for all tokens in text.

    Values are raw term frequencies. IDF is applied server-side by Qdrant via
    modifier: idf on the sparse space declaration. Never compute IDF here —
    doing so would double-apply it and corrupt BM25 scores.

    Returns an empty dict for empty or whitespace-only input.
    """
    counts = Counter(bm25_tokenize(text))
    return {stable_u32_hash(token): float(count) for token, count in counts.items()}


def to_qdrant_sparse(text: str) -> SparseVector:
    """Return a Qdrant SparseVector encoding the BM25 term frequencies for text.

    Indices are sorted ascending — a Qdrant requirement for sparse vectors.
    This is a format adapter over build_sparse_vector with no additional logic.

    Use for:
        PointStruct vector field on bm25_code / bm25_paper spaces (upsert)
        SparseVector argument in Query API prefetch legs (retrieval)
    """
    pairs = sorted(build_sparse_vector(text).items())
    return SparseVector(
        indices=[idx for idx, _ in pairs],
        values=[val for _, val in pairs],
    )