"""Tests for tokenizer.py

Test classes mirror the module's public functions. Run with: pytest tests/test_tokenizer.py -v

The BM25BugRegression class at the bottom documents a historical failure mode
and verifies it stays fixed. Do not delete or weaken those tests — they exist
to prevent silent regressions to dead sparse retrieval.
"""

from collections import Counter

import pytest

from kernelpack_rag.tokenizer import (
    bm25_tokenize,
    build_sparse_vector,
    lexical_split,
    split_dots,
    split_snake_camel,
    stable_u32_hash,
    tokenize,
)


# ---------------------------------------------------------------------------
# lexical_split
# ---------------------------------------------------------------------------

class TestLexicalSplit:
    """lexical_split preserves dotted names and strips punctuation/whitespace."""

    def test_simple_word(self):
        assert list(lexical_split("alpha")) == ["alpha"]

    def test_dotted_name_preserved_as_one_token(self):
        assert list(lexical_split("PoissonSolver.solve")) == ["PoissonSolver.solve"]

    def test_multi_segment_dotted_name(self):
        assert list(lexical_split("kernelpack.solvers.PoissonSolver")) == [
            "kernelpack.solvers.PoissonSolver"
        ]

    def test_whitespace_splits_tokens(self):
        assert list(lexical_split("foo bar baz")) == ["foo", "bar", "baz"]

    def test_parens_and_commas_stripped(self):
        result = list(lexical_split("solve(A, b)"))
        assert result == ["solve", "A", "b"]

    def test_type_annotation_line(self):
        # Realistic docstring fragment — colons and arrows are not identifiers
        result = list(lexical_split("def solve(self, A: np.ndarray) -> np.ndarray:"))
        assert "solve" in result
        assert "self" in result
        assert "A" in result
        assert "np.ndarray" in result
        assert "->" not in result

    def test_underscore_is_part_of_identifier(self):
        # Underscores do NOT split at this stage — that is split_snake_camel's job
        assert list(lexical_split("compute_stencil")) == ["compute_stencil"]

    def test_numbers_kept(self):
        assert list(lexical_split("bdf3")) == ["bdf3"]

    def test_empty_string(self):
        assert list(lexical_split("")) == []

    def test_whitespace_only(self):
        assert list(lexical_split("   \t\n")) == []

    def test_punctuation_only(self):
        assert list(lexical_split("()[]{}")) == []

    def test_multiple_dotted_names(self):
        result = list(lexical_split("RBFStencil.assemble rbf_fd.solve"))
        assert "RBFStencil.assemble" in result
        assert "rbf_fd.solve" in result


# ---------------------------------------------------------------------------
# split_dots
# ---------------------------------------------------------------------------

class TestSplitDots:
    """split_dots splits on dots and omits empty parts."""

    def test_no_dots_returns_whole(self):
        assert list(split_dots("alpha")) == ["alpha"]

    def test_single_dot(self):
        assert list(split_dots("PoissonSolver.solve")) == ["PoissonSolver", "solve"]

    def test_three_segments(self):
        assert list(split_dots("kernelpack.solvers.PoissonSolver")) == [
            "kernelpack", "solvers", "PoissonSolver"
        ]

    def test_empty_parts_omitted(self):
        # Guard against malformed input — empty strings must not appear
        result = list(split_dots(".foo."))
        assert "" not in result
        assert "foo" in result


# ---------------------------------------------------------------------------
# split_snake_camel
# ---------------------------------------------------------------------------

class TestSplitSnakeCamel:
    """split_snake_camel decomposes identifiers into lowercase atomic fragments."""

    def test_snake_case(self):
        assert list(split_snake_camel("compute_stencil")) == ["compute", "stencil"]

    def test_camel_case(self):
        assert list(split_snake_camel("PoissonSolver")) == ["poisson", "solver"]

    def test_all_caps_acronym_at_end(self):
        # Regex handles trailing all-caps via the $ anchor in the lookahead
        assert list(split_snake_camel("RBF")) == ["rbf"]

    def test_acronym_prefix_before_camel(self):
        # RBFStencil → rbf, stencil
        assert list(split_snake_camel("RBFStencil")) == ["rbf", "stencil"]

    def test_acronym_prefix_fd(self):
        # FDDiffOp → fd, diff, op
        assert list(split_snake_camel("FDDiffOp")) == ["fd", "diff", "op"]

    def test_snake_then_camel(self):
        result = list(split_snake_camel("compute_RBFStencil"))
        assert "compute" in result
        assert "rbf" in result
        assert "stencil" in result

    def test_leading_underscore(self):
        assert list(split_snake_camel("_common")) == ["common"]

    def test_double_leading_underscore(self):
        # __init__ → init
        assert list(split_snake_camel("__init__")) == ["init"]

    def test_trailing_underscore(self):
        result = list(split_snake_camel("solve_"))
        assert "solve" in result
        assert "" not in result

    def test_all_lowercase_passthrough(self):
        assert list(split_snake_camel("solve")) == ["solve"]

    def test_digits_split(self):
        result = list(split_snake_camel("bdf3"))
        assert "bdf" in result
        assert "3" in result

    def test_digit_letter_suffix(self):
        # "2d" → ["2", "d"]
        result = list(split_snake_camel("rbf_fd_2d"))
        assert "rbf" in result
        assert "fd" in result
        assert "2" in result
        assert "d" in result

    def test_long_camel(self):
        result = list(split_snake_camel("NonlinearVariablePoissonSolver"))
        assert result == ["nonlinear", "variable", "poisson", "solver"]

    def test_empty_string(self):
        assert list(split_snake_camel("")) == []


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    """tokenize: multi-level emission with per-source-token deduplication.

    The invariant: each distinct string appears AT MOST ONCE per source lexical
    token. TF reflects how many times a token string appears across the full
    text, not how many times the emission loop yields it for a single occurrence.
    """

    def test_dotted_identifier_all_levels_present(self):
        tokens = list(tokenize("PoissonSolver.solve"))
        assert "poissonsolver.solve" in tokens   # whole dotted token
        assert "poissonsolver" in tokens          # dot-part
        assert "poisson" in tokens                # camel fragment
        assert "solver" in tokens                 # camel fragment
        assert "solve" in tokens                  # dot-part (also a fragment)

    def test_no_duplicate_within_single_source_token_simple(self):
        # "alpha" has no dots and no camel/snake structure.
        # Must appear exactly once, not 2 or 3 times.
        tokens = list(tokenize("alpha"))
        assert tokens.count("alpha") == 1

    def test_no_duplicate_within_single_source_token_dotpart(self):
        # "solve" is emitted as a dot-part AND what split_snake_camel returns.
        # Must appear exactly once per source token.
        tokens = list(tokenize("PoissonSolver.solve"))
        assert tokens.count("solve") == 1
        assert tokens.count("poissonsolver") == 1

    def test_tf_accumulates_correctly_across_source_tokens(self):
        # "alpha" appears twice in the text → TF should be 2, not 6.
        counts = Counter(tokenize("alpha alpha"))
        assert counts["alpha"] == 2

    def test_three_repeated_tokens(self):
        counts = Counter(tokenize("solve solve solve"))
        assert counts["solve"] == 3

    def test_snake_case_all_levels(self):
        tokens = list(tokenize("compute_stencil_laplacian"))
        assert "compute_stencil_laplacian" in tokens  # whole
        assert "compute" in tokens
        assert "stencil" in tokens
        assert "laplacian" in tokens

    def test_multiword_text(self):
        tokens = list(tokenize("how does PoissonSolver work"))
        assert "how" in tokens
        assert "does" in tokens
        assert "poissonsolver" in tokens
        assert "poisson" in tokens
        assert "solver" in tokens
        assert "work" in tokens

    def test_all_caps_acronym(self):
        tokens = list(tokenize("RBFStencil"))
        assert "rbfstencil" in tokens   # whole
        assert "rbf" in tokens
        assert "stencil" in tokens

    def test_numbers_in_identifier(self):
        tokens = list(tokenize("bdf3"))
        assert "bdf3" in tokens         # whole
        assert "bdf" in tokens
        assert "3" in tokens

    def test_empty_string(self):
        assert list(tokenize("")) == []

    def test_whitespace_only(self):
        assert list(tokenize("   ")) == []

    def test_punctuation_only(self):
        assert list(tokenize("()[]{}")) == []

    def test_realistic_function_signature(self):
        text = "def assemble_op(self, domain: DomainDescriptor) -> np.ndarray:"
        tokens = list(tokenize(text))
        assert "assemble_op" in tokens
        assert "assemble" in tokens
        assert "op" in tokens
        assert "domaindescriptor" in tokens
        assert "domain" in tokens
        assert "descriptor" in tokens


# ---------------------------------------------------------------------------
# bm25_tokenize
# ---------------------------------------------------------------------------

class TestBm25Tokenize:
    """bm25_tokenize wraps tokenize as a list. TF = count of each string."""

    def test_returns_list(self):
        assert isinstance(bm25_tokenize("alpha"), list)

    def test_correct_tf_two_occurrences(self):
        counts = Counter(bm25_tokenize("alpha alpha beta"))
        assert counts["alpha"] == 2
        assert counts["beta"] == 1

    def test_no_overcounting_within_source_token(self):
        # Regression guard: the overcounting bug gave alpha TF=3 for one occurrence.
        counts = Counter(bm25_tokenize("alpha"))
        assert counts["alpha"] == 1

    def test_dotted_identifier_single_occurrence(self):
        counts = Counter(bm25_tokenize("PoissonSolver.solve"))
        assert counts["solve"] == 1
        assert counts["poisson"] == 1
        assert counts["solver"] == 1
        assert counts["poissonsolver"] == 1
        assert counts["poissonsolver.solve"] == 1


# ---------------------------------------------------------------------------
# stable_u32_hash
# ---------------------------------------------------------------------------

class TestStableU32Hash:
    """stable_u32_hash is deterministic, collision-resistant at project scale,
    and always in [0, 2**32)."""

    def test_deterministic_same_input(self):
        assert stable_u32_hash("laplacian") == stable_u32_hash("laplacian")

    def test_in_u32_range(self):
        for token in ["poisson", "solve", "rbf", "stencil", "", "0", "__init__"]:
            h = stable_u32_hash(token)
            assert 0 <= h < 2 ** 32, f"Hash out of range for token: {repr(token)}"

    def test_no_collisions_in_domain_vocabulary(self):
        # Not theoretically guaranteed, but must hold for the project vocabulary.
        # A collision here would corrupt BM25 term weights silently.
        vocab = [
            "laplacian", "poisson", "stencil", "rbf", "solve", "alpha", "beta",
            "phs", "bdf3", "neumann", "dirichlet", "jacobian", "gmres",
            "variablepoissonsolver", "assemble", "domain", "descriptor",
        ]
        hashes = [stable_u32_hash(t) for t in vocab]
        assert len(set(hashes)) == len(hashes), (
            "Hash collision in domain vocabulary — BM25 term weights will be wrong"
        )

    def test_empty_string_does_not_raise(self):
        h = stable_u32_hash("")
        assert 0 <= h < 2 ** 32

    def test_different_tokens_produce_different_hashes(self):
        assert stable_u32_hash("solve") != stable_u32_hash("solver")
        assert stable_u32_hash("poisson") != stable_u32_hash("poissonsolver")


# ---------------------------------------------------------------------------
# build_sparse_vector
# ---------------------------------------------------------------------------

class TestBuildSparseVector:
    """build_sparse_vector: {hash → raw TF (float)}.

    NOTE: this function is here temporarily. It moves to embed/sparse.py at
    Step 11. When it moves, move these tests to tests/test_sparse.py.
    """

    def test_basic_tf(self):
        vec = build_sparse_vector("alpha alpha beta")
        assert vec[stable_u32_hash("alpha")] == 2.0
        assert vec[stable_u32_hash("beta")] == 1.0

    def test_values_are_float(self):
        vec = build_sparse_vector("alpha")
        for v in vec.values():
            assert isinstance(v, float)

    def test_empty_string_returns_empty_dict(self):
        assert build_sparse_vector("") == {}

    def test_dotted_identifier_tf_per_token(self):
        # Each distinct emitted token appears once from a single source token
        vec = build_sparse_vector("PoissonSolver.solve")
        assert vec[stable_u32_hash("poissonsolver.solve")] == 1.0
        assert vec[stable_u32_hash("poisson")] == 1.0
        assert vec[stable_u32_hash("solver")] == 1.0
        assert vec[stable_u32_hash("solve")] == 1.0

    def test_tf_accumulates_across_repeated_source_tokens(self):
        vec = build_sparse_vector("solve solve solve")
        assert vec[stable_u32_hash("solve")] == 3.0


# ---------------------------------------------------------------------------
# BM25 bug regression
# ---------------------------------------------------------------------------

class TestBm25BugRegression:
    """Documents the historical dead-sparse-leg bug and verifies it stays fixed.

    WHAT BROKE:
    The original tokenizer used whitespace .split(). Indexing the chunk text
    'VariablePoissonSolver.solve' produced a single token: the full dotted string.
    A natural-language query 'VariablePoissonSolver solve' (space-separated)
    produced 'VariablePoissonSolver' — note: NO dot. These two strings never
    matched, so BM25 scored every chunk near-zero. The sparse retrieval leg was
    functionally dead. Hybrid search silently degraded to dense-only with RRF
    overhead.

    WHAT THE FIX DOES:
    Both the indexed chunk and the query emit shared fragments ('variablepoissonsolver',
    'variable', 'poisson', 'solver', 'solve'). BM25 now scores genuine signal.

    Do not delete or weaken these tests.
    """

    def test_dotted_and_space_separated_share_fragments(self):
        """Core regression: index has dotted form, query has space-separated form."""
        index_tokens = set(bm25_tokenize("VariablePoissonSolver.solve"))
        query_tokens = set(bm25_tokenize("VariablePoissonSolver solve"))
        overlap = index_tokens & query_tokens
        assert len(overlap) > 0, "Zero token overlap — sparse leg is dead"
        assert "variablepoissonsolver" in overlap
        assert "solve" in overlap

    def test_compound_camel_identifier_fully_decomposed(self):
        """Old tokenizer treated the whole compound as one atomic token."""
        tokens = set(bm25_tokenize("NonlinearVariablePoissonSolver.solve"))
        assert "nonlinear" in tokens
        assert "variable" in tokens
        assert "poisson" in tokens
        assert "solver" in tokens
        assert "solve" in tokens

    def test_natural_language_query_matches_code_identifier(self):
        """Plain English query should share tokens with the indexed method name."""
        query = set(bm25_tokenize("how does the poisson solver solve the system"))
        index = set(bm25_tokenize("PoissonSolver.solve"))
        shared = query & index
        assert "poisson" in shared
        assert "solver" in shared
        assert "solve" in shared

    def test_rbf_acronym_query_matches_index(self):
        query = set(bm25_tokenize("RBF stencil assembly"))
        index = set(bm25_tokenize("RBFStencil.assemble"))
        shared = query & index
        assert "rbf" in shared
        assert "stencil" in shared

    def test_snake_case_query_matches_code(self):
        """snake_case in source should match fragments in a query."""
        query = set(bm25_tokenize("compute laplacian stencil"))
        index = set(bm25_tokenize("compute_stencil_laplacian"))
        shared = query & index
        assert "compute" in shared
        assert "stencil" in shared
        assert "laplacian" in shared

    def test_what_broken_tokenizer_produced(self):
        """Shows exactly why the old tokenizer failed.
        This is NOT testing the fixed tokenizer — it demonstrates the bug.
        The whitespace split produces zero overlap between dotted index token
        and space-separated query token. The tests above verify the fix."""
        broken_index = "VariablePoissonSolver.solve".split()
        broken_query = "VariablePoissonSolver solve".split()
        assert set(broken_index) & set(broken_query) == set(), (
            "This documents the bug: broken tokenizer has zero overlap. "
            "The fixed tokenizer should pass the tests above instead."
        )