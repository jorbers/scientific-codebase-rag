"""Tests for the fine chunker (Step 6).

Interface:
    fine_chunks(coarse: CoarseChunk) -> list[FineChunk]

FineChunk fields:
    text: str                    — window text, no trailing newline
    qualname: str                — inherited from CoarseChunk
    chunk_type: str              — always "fine"
    line_range: tuple[int, int]  — absolute 1-indexed line numbers from the source file
    source_file: str             — inherited from CoarseChunk
    parent_class: str | None     — inherited from CoarseChunk
    parent_id: uuid.UUID         — uuid5(KP_NAMESPACE, f"code:{source_file}:{qualname}:coarse:0")
    window_idx: int              — 0-indexed, sequential across windows
    granularity: str             — always "fine"

Windowing rules:
    - Split only at tree-sitter statement boundaries (never mid-expression).
    - Target window size: 3–5 lines.
    - Every window except the last must be >= 3 lines.
    - A statement that alone exceeds 5 lines becomes its own window (not further split).
    - Windows are non-overlapping. Together they cover every line of the coarse chunk.
    - The def/async def signature line (plus any decorators) is part of window 0, not a
      standalone window.

Line range rules:
    - line_range is derived from the parent's line_range[0] as an offset.
    - first window: line_range[0] == coarse.line_range[0]
    - last window:  line_range[1] == coarse.line_range[1]
    - consecutive windows are contiguous: window[i].line_range[1] + 1 == window[i+1].line_range[0]

Chunk type filter:
    - Returns [] for chunk_type in {"class_header", "module_docstring"}.
    - Processes chunk_type in {"function", "method"}.

ID rules (D5):
    parent_id = uuid5(KP_NAMESPACE, f"code:{coarse.source_file}:{coarse.qualname}:coarse:0")
    KP_NAMESPACE is a module-level uuid.UUID constant exported from fine.py.
"""

import uuid

import pytest

from kernelpack_rag.chunking.coarse import CoarseChunk
from kernelpack_rag.chunking.fine import FineChunk, KP_NAMESPACE, fine_chunks


# ── fixtures ──────────────────────────────────────────────────────────────────

# 9-line method — long enough to force multiple windows
FUNC_9 = """\
def solve(self, rhs, tol):
    A = self._build_matrix()
    b = self._build_rhs(rhs)
    x = np.zeros_like(rhs)
    norm = np.linalg.norm(b)
    scaled = b / norm
    result = A @ scaled
    out = result * tol
    return out"""

# 5-line method — right at the coarse MIN_LINES boundary
FUNC_5 = """\
def helper(self, x, y):
    a = x + y
    b = a * 2
    c = b - x
    return c"""

# 8-line function with a multi-line call expression spanning lines 2–6
# The call must not be split mid-expression.
FUNC_MULTILINE_CALL = """\
def build(self, config, opts):
    result = some_function(
        arg1=config,
        arg2=opts,
        arg3=self.default,
    )
    out = result * 2
    return out"""

# 7-line standalone function (no parent class)
FUNC_STANDALONE = """\
def compute(x, y, z):
    a = x + y
    b = a * z
    c = b - x
    d = c / y
    e = d + z
    return e"""


def make_coarse(
    text: str,
    qualname: str = "Solver.solve",
    chunk_type: str = "method",
    line_range: tuple[int, int] | None = None,
    source_file: str = "solver.py",
    parent_class: str | None = "Solver",
) -> CoarseChunk:
    if line_range is None:
        n = len(text.splitlines())
        line_range = (1, n)
    return CoarseChunk(
        text=text,
        qualname=qualname,
        chunk_type=chunk_type,
        line_range=line_range,
        source_file=source_file,
        parent_class=parent_class,
    )


# ── return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_list(self):
        coarse = make_coarse(FUNC_5)
        assert isinstance(fine_chunks(coarse), list)

    def test_elements_are_fine_chunks(self):
        coarse = make_coarse(FUNC_9)
        result = fine_chunks(coarse)
        assert len(result) > 0
        assert all(isinstance(fc, FineChunk) for fc in result)

    def test_non_empty_for_function(self):
        coarse = make_coarse(FUNC_5)
        assert len(fine_chunks(coarse)) >= 1

    def test_non_empty_for_method(self):
        coarse = make_coarse(FUNC_9, chunk_type="method")
        assert len(fine_chunks(coarse)) >= 1


# ── field values ─────────────────────────────────────────────────────────────

class TestFields:
    def test_granularity_is_fine(self):
        for fc in fine_chunks(make_coarse(FUNC_9)):
            assert fc.granularity == "fine"

    def test_chunk_type_is_fine(self):
        for fc in fine_chunks(make_coarse(FUNC_9)):
            assert fc.chunk_type == "fine"

    def test_qualname_inherited(self):
        coarse = make_coarse(FUNC_9, qualname="MySolver.compute")
        for fc in fine_chunks(coarse):
            assert fc.qualname == "MySolver.compute"

    def test_source_file_inherited(self):
        coarse = make_coarse(FUNC_9, source_file="kernelpack/solvers.py")
        for fc in fine_chunks(coarse):
            assert fc.source_file == "kernelpack/solvers.py"

    def test_parent_class_inherited(self):
        coarse = make_coarse(FUNC_9, parent_class="PoissonSolver")
        for fc in fine_chunks(coarse):
            assert fc.parent_class == "PoissonSolver"

    def test_parent_class_none_inherited(self):
        coarse = make_coarse(
            FUNC_STANDALONE,
            qualname="compute",
            chunk_type="function",
            parent_class=None,
        )
        for fc in fine_chunks(coarse):
            assert fc.parent_class is None

    def test_window_idx_sequential_from_zero(self):
        result = fine_chunks(make_coarse(FUNC_9))
        for i, fc in enumerate(result):
            assert fc.window_idx == i

    def test_text_is_str(self):
        for fc in fine_chunks(make_coarse(FUNC_5)):
            assert isinstance(fc.text, str)

    def test_text_not_empty(self):
        for fc in fine_chunks(make_coarse(FUNC_9)):
            assert fc.text.strip() != ""


# ── parent ID ─────────────────────────────────────────────────────────────────

class TestParentId:
    def test_parent_id_is_uuid(self):
        for fc in fine_chunks(make_coarse(FUNC_5)):
            assert isinstance(fc.parent_id, uuid.UUID)

    def test_parent_id_matches_d5_formula(self):
        coarse = make_coarse(FUNC_5, qualname="Solver.solve", source_file="solver.py")
        expected = uuid.uuid5(KP_NAMESPACE, "code:solver.py:Solver.solve:coarse:0")
        for fc in fine_chunks(coarse):
            assert fc.parent_id == expected

    def test_parent_id_varies_with_qualname(self):
        coarse_a = make_coarse(FUNC_5, qualname="A.solve", source_file="a.py")
        coarse_b = make_coarse(FUNC_5, qualname="B.solve", source_file="b.py")
        id_a = fine_chunks(coarse_a)[0].parent_id
        id_b = fine_chunks(coarse_b)[0].parent_id
        assert id_a != id_b

    def test_parent_id_consistent_across_windows(self):
        result = fine_chunks(make_coarse(FUNC_9))
        ids = {fc.parent_id for fc in result}
        assert len(ids) == 1


# ── windowing ─────────────────────────────────────────────────────────────────

class TestWindowing:
    def test_multiple_windows_for_long_function(self):
        assert len(fine_chunks(make_coarse(FUNC_9))) >= 2

    def test_window_size_at_least_3_except_last(self):
        result = fine_chunks(make_coarse(FUNC_9))
        for fc in result[:-1]:
            line_count = fc.line_range[1] - fc.line_range[0] + 1
            assert line_count >= 3, (
                f"Window {fc.window_idx} has {line_count} lines — expected >= 3 for non-last window"
            )

    def test_window_size_at_most_5_no_multiline_expr(self):
        # FUNC_9 has no multi-line expressions; all windows must be <= 5 lines
        result = fine_chunks(make_coarse(FUNC_9))
        for fc in result:
            line_count = fc.line_range[1] - fc.line_range[0] + 1
            assert line_count <= 5, (
                f"Window {fc.window_idx} has {line_count} lines — expected <= 5"
            )

    def test_multiline_expression_not_split(self):
        # some_function(...) occupies lines 2–6 in FUNC_MULTILINE_CALL.
        # No window boundary should fall inside lines 2–5 (mid-statement).
        coarse = make_coarse(FUNC_MULTILINE_CALL)
        result = fine_chunks(coarse)
        mid_call_lines = {2, 3, 4, 5}  # line 6 is the closing paren — a valid boundary
        for fc in result:
            assert fc.line_range[1] not in mid_call_lines, (
                f"Window {fc.window_idx} ends at line {fc.line_range[1]}, "
                f"splitting inside the multi-line call expression (lines 2–6)"
            )

    def test_large_statement_emitted_as_own_window(self):
        # The 5-line call expression in FUNC_MULTILINE_CALL must appear
        # intact in exactly one window.
        coarse = make_coarse(FUNC_MULTILINE_CALL)
        result = fine_chunks(coarse)
        # Find the window that contains line 2 (start of the call)
        containing = [fc for fc in result if fc.line_range[0] <= 2 <= fc.line_range[1]]
        assert len(containing) == 1
        # That same window must also contain line 6 (end of the call)
        assert containing[0].line_range[1] >= 6


# ── line range coverage ───────────────────────────────────────────────────────

class TestLineRange:
    def test_first_window_starts_at_parent_start(self):
        coarse = make_coarse(FUNC_9, line_range=(10, 18))
        result = fine_chunks(coarse)
        assert result[0].line_range[0] == 10

    def test_last_window_ends_at_parent_end(self):
        coarse = make_coarse(FUNC_9, line_range=(10, 18))
        result = fine_chunks(coarse)
        assert result[-1].line_range[1] == 18

    def test_windows_are_contiguous(self):
        coarse = make_coarse(FUNC_9, line_range=(10, 18))
        result = fine_chunks(coarse)
        for i in range(len(result) - 1):
            assert result[i].line_range[1] + 1 == result[i + 1].line_range[0], (
                f"Gap between window {i} (ends {result[i].line_range[1]}) "
                f"and window {i+1} (starts {result[i+1].line_range[0]})"
            )

    def test_line_range_start_lte_end(self):
        for fc in fine_chunks(make_coarse(FUNC_9)):
            assert fc.line_range[0] <= fc.line_range[1]

    def test_non_default_parent_offset(self):
        # Verify offset arithmetic: parent starts at line 50
        coarse = make_coarse(FUNC_5, line_range=(50, 54))
        result = fine_chunks(coarse)
        assert result[0].line_range[0] == 50
        assert result[-1].line_range[1] == 54


# ── chunk type filter ─────────────────────────────────────────────────────────

class TestChunkTypeFilter:
    def test_class_header_returns_empty(self):
        coarse = make_coarse(
            text="class Solver:\n    pass",
            qualname="Solver",
            chunk_type="class_header",
            parent_class=None,
        )
        assert fine_chunks(coarse) == []

    def test_module_docstring_returns_empty(self):
        coarse = make_coarse(
            text='"""Module docstring."""',
            qualname="solver",
            chunk_type="module_docstring",
            parent_class=None,
        )
        assert fine_chunks(coarse) == []

    def test_function_chunk_processed(self):
        coarse = make_coarse(FUNC_STANDALONE, chunk_type="function", parent_class=None)
        assert len(fine_chunks(coarse)) >= 1

    def test_method_chunk_processed(self):
        coarse = make_coarse(FUNC_9, chunk_type="method")
        assert len(fine_chunks(coarse)) >= 1