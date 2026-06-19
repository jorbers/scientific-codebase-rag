"""Tests for embed/sparse.py — sparse BM25 vector builder.

Expected public interface of embed/sparse.py:

    build_sparse_vector(text: str) -> dict[int, float]
        Maps stable_u32_hash(token) -> raw_TF for all tokens in text.
        Raw term frequencies only. IDF is NOT applied client-side —
        Qdrant handles it server-side via modifier: idf on bm25_code / bm25_paper.

    to_qdrant_sparse(text: str) -> qdrant_client.models.SparseVector
        Qdrant-ready wrapper. Indices sorted ascending (Qdrant requirement).
        Encodes the same TF data as build_sparse_vector — no new logic.

Run with: pytest tests/test_sparse.py -v
"""

from __future__ import annotations

import pytest
from qdrant_client.models import SparseVector

from embed.sparse import build_sparse_vector, to_qdrant_sparse
from kernelpack_rag.tokenizer import stable_u32_hash


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key(token: str) -> int:
    """Shorthand: the expected dict key for a given token string."""
    return stable_u32_hash(token)


# ---------------------------------------------------------------------------
# build_sparse_vector
# ---------------------------------------------------------------------------

class TestBuildSparseVector:

    def test_returns_dict(self):
        assert isinstance(build_sparse_vector("PoissonSolver.solve"), dict)

    def test_keys_are_nonneg_u32_ints(self):
        result = build_sparse_vector("PoissonSolver.solve")
        for k in result:
            assert isinstance(k, int)
            assert 0 <= k < 2**32

    def test_values_are_floats(self):
        result = build_sparse_vector("PoissonSolver.solve")
        for v in result.values():
            assert isinstance(v, float)

    def test_empty_input(self):
        assert build_sparse_vector("") == {}

    # --- TF accumulation ---

    def test_tf_accumulates_across_source_tokens(self):
        """'solve' appears as a standalone token 3×; TF must be 3.0."""
        result = build_sparse_vector("solve solve solve")
        assert result[_key("solve")] == 3.0

    def test_tf_accumulates_for_shared_fragment(self):
        """'solve' is a dot-segment fragment in two separate identifiers.
        TF should be 2.0, not 1.0."""
        result = build_sparse_vector("PoissonSolver.solve QuadraticSolver.solve")
        assert result[_key("solve")] == 2.0

    # --- Per-token deduplication ---

    def test_dedup_within_single_identifier(self):
        """In 'PoissonSolver.solve', 'solve' is emitted as a candidate twice:
        once from split_dots and once from split_snake_camel('solve').
        Per-token dedup means TF stays 1.0, not 2.0."""
        result = build_sparse_vector("PoissonSolver.solve")
        assert result[_key("solve")] == 1.0

    # --- Known token coverage ---

    def test_known_identifier_all_expected_tokens_present(self):
        """'PoissonSolver.solve' must produce all five expected tokens.
        Any missing token means a query phrasing will silently miss this chunk."""
        result = build_sparse_vector("PoissonSolver.solve")
        expected = [
            "poissonsolver.solve",  # whole lowercased — exact-lookup leg
            "poissonsolver",        # dot-segment
            "poisson",              # camelCase fragment
            "solver",               # camelCase fragment
            "solve",                # dot-segment / method name
        ]
        for tok in expected:
            assert _key(tok) in result, f"Token '{tok}' missing from sparse vector"

    def test_snake_case_identifier_fragments_present(self):
        """'compute_laplacian' must produce 'compute' and 'laplacian' as fragments
        so NL queries like 'laplacian operator' hit chunks using snake_case names."""
        result = build_sparse_vector("compute_laplacian")
        assert _key("compute") in result
        assert _key("laplacian") in result

    # --- No IDF ---

    def test_no_idf_applied(self):
        """Values must be raw TF. IDF is applied server-side by Qdrant.
        Computing it client-side would double-apply it and corrupt scores."""
        result = build_sparse_vector("solve solve")
        # If IDF were folded in, this would not be 2.0.
        assert result[_key("solve")] == 2.0

    # --- Determinism ---

    def test_deterministic(self):
        text = "RBFStencil.assemble_op compute_laplacian phs"
        assert build_sparse_vector(text) == build_sparse_vector(text)


# ---------------------------------------------------------------------------
# to_qdrant_sparse
# ---------------------------------------------------------------------------

class TestToQdrantSparse:

    def test_returns_sparse_vector(self):
        result = to_qdrant_sparse("PoissonSolver.solve")
        assert isinstance(result, SparseVector)

    def test_indices_sorted_ascending(self):
        """Qdrant requires indices in ascending order for sparse vectors."""
        result = to_qdrant_sparse("PoissonSolver.solve alpha beta gamma laplacian")
        assert result.indices == sorted(result.indices)

    def test_indices_and_values_same_length(self):
        result = to_qdrant_sparse("PoissonSolver.solve")
        assert len(result.indices) == len(result.values)

    def test_empty_input(self):
        result = to_qdrant_sparse("")
        assert isinstance(result, SparseVector)
        assert result.indices == []
        assert result.values == []

    def test_encodes_same_data_as_build_sparse_vector(self):
        """to_qdrant_sparse is a format wrapper — same TF data, different shape.
        Reconstructing a dict from the SparseVector must equal build_sparse_vector."""
        text = "RBFStencil compute_laplacian phs"
        vec_dict = build_sparse_vector(text)
        sv = to_qdrant_sparse(text)
        reconstructed = dict(zip(sv.indices, sv.values))
        assert reconstructed == vec_dict

    def test_no_duplicate_indices(self):
        """Each token hashes to one index. Duplicates would corrupt TF values."""
        result = to_qdrant_sparse("PoissonSolver.solve compute_laplacian rbf phs")
        assert len(result.indices) == len(set(result.indices))


# ---------------------------------------------------------------------------
# BM25 regression — documents the historical whitespace-split failure
# ---------------------------------------------------------------------------

class TestBM25BugRegression:
    """The original pipeline used str.split() for tokenization.
    'VariablePoissonSolver.solve' became one token that never matched any
    natural-language or space-separated query. This class pins that fix.
    Do not delete or weaken these tests."""

    def test_dotted_identifier_produces_fragments(self):
        """With str.split(), 'VariablePoissonSolver.solve' → one token.
        Correct: must produce fragment tokens that overlap with NL queries."""
        result = build_sparse_vector("VariablePoissonSolver.solve")
        assert _key("variable") in result
        assert _key("poisson") in result
        assert _key("solver") in result
        assert _key("solve") in result

    def test_nl_query_tokens_overlap_with_identifier_tokens(self):
        """A query 'variable poisson solver' (NL phrasing) must share token
        hashes with a chunk containing 'VariablePoissonSolver'. Without the
        identifier-aware tokenizer, BM25 score is zero."""
        chunk_vec = build_sparse_vector("VariablePoissonSolver.solve")
        query_vec = build_sparse_vector("variable poisson solver solve")
        overlap = set(chunk_vec.keys()) & set(query_vec.keys())
        assert len(overlap) >= 3, (
            f"Expected ≥3 overlapping tokens, got {len(overlap)}: {overlap}"
        )