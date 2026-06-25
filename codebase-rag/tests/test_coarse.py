"""Tests for the coarse chunker (Step 4).

Conventions enforced here:
- line_range is 1-indexed
- Every function and method is emitted as its own chunk regardless of size
- Short methods are excluded from the class_header text (same as long methods)
"""
import textwrap
from pathlib import Path
import pytest
from kernelpack_rag.chunking.coarse import CoarseChunk, chunk_file

VALID_CHUNK_TYPES = frozenset({"function", "method", "class_header", "module_docstring"})


def write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(src))
    return p


# ── dataclass fields ──────────────────────────────────────────────────────────

class TestCoarseChunkFields:
    def test_required_fields_exist(self):
        chunk = CoarseChunk(
            text="def f(x):\n    return x",
            qualname="f",
            chunk_type="function",
            line_range=(1, 2),
            source_file="mod.py",
            parent_class=None,
            module="kernelpack.test"
        )
        assert chunk.text == "def f(x):\n    return x"
        assert chunk.qualname == "f"
        assert chunk.chunk_type == "function"
        assert chunk.line_range == (1, 2)
        assert chunk.source_file == "mod.py"
        assert chunk.parent_class is None


# ── standalone functions ──────────────────────────────────────────────────────

class TestStandaloneFunction:
    # 5 lines — exactly at MIN_LINES, must be extracted
    SRC = """\
        def solve(x, y, z):
            a = x + y
            b = a * z
            c = b - 1
            return c
    """

    def test_extracted(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        funcs = [c for c in chunk_file(p) if c.chunk_type == "function"]
        assert len(funcs) == 1

    def test_qualname(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        funcs = [c for c in chunk_file(p) if c.chunk_type == "function"]
        assert funcs[0].qualname == "solve"

    def test_parent_class_is_none(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        funcs = [c for c in chunk_file(p) if c.chunk_type == "function"]
        assert funcs[0].parent_class is None

    def test_source_file(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        funcs = [c for c in chunk_file(p) if c.chunk_type == "function"]
        assert funcs[0].source_file == str(p)

    def test_line_range(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        funcs = [c for c in chunk_file(p) if c.chunk_type == "function"]
        assert funcs[0].line_range == (1, 5)

    def test_short_standalone_function_indexed(self, tmp_path):
        # 2 lines — must be indexed, not dropped
        src = """\
            def tiny(x):
                pass
        """
        p = write(tmp_path, "mod.py", src)
        funcs = [c for c in chunk_file(p) if c.chunk_type == "function"]
        assert len(funcs) == 1


# ── class and methods ─────────────────────────────────────────────────────────

class TestClassAndMethods:
    # solve = 6 lines, tiny = 2 lines — both emitted separately, both excluded from class_header
    SRC = """\
        class PoissonSolver:
            \"\"\"Solves Poisson equations.\"\"\"

            def solve(self, rhs, domain, tol):
                result = self._compute(rhs)
                scaled = result * domain
                return scaled / tol

            def tiny(self):
                pass
    """

    def test_class_header_emitted(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        headers = [c for c in chunk_file(p) if c.chunk_type == "class_header"]
        assert len(headers) == 1

    def test_class_header_qualname(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        headers = [c for c in chunk_file(p) if c.chunk_type == "class_header"]
        assert headers[0].qualname == "PoissonSolver"

    def test_class_header_no_parent_class(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        headers = [c for c in chunk_file(p) if c.chunk_type == "class_header"]
        assert headers[0].parent_class is None

    def test_both_methods_emitted(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        methods = [c for c in chunk_file(p) if c.chunk_type == "method"]
        assert len(methods) == 2

    def test_method_qualnames(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        methods = [c for c in chunk_file(p) if c.chunk_type == "method"]
        qualnames = {c.qualname for c in methods}
        assert qualnames == {"PoissonSolver.solve", "PoissonSolver.tiny"}

    def test_method_parent_class(self, tmp_path):
        p = write(tmp_path, "mod.py", self.SRC)
        methods = [c for c in chunk_file(p) if c.chunk_type == "method"]
        assert all(c.parent_class == "PoissonSolver" for c in methods)

    def test_short_method_excluded_from_class_header(self, tmp_path):
        # All methods are excluded from class_header regardless of size
        p = write(tmp_path, "mod.py", self.SRC)
        headers = [c for c in chunk_file(p) if c.chunk_type == "class_header"]
        assert "def tiny" not in headers[0].text
        assert "def solve" not in headers[0].text


# ── module docstring ──────────────────────────────────────────────────────────

class TestModuleDocstring:
    def test_emitted_when_present(self, tmp_path):
        src = """\
            \"\"\"Module for solving Poisson problems.\"\"\"

            def solve(x, y, z):
                a = x + y
                b = a * z
                return b
        """
        p = write(tmp_path, "mod.py", src)
        docs = [c for c in chunk_file(p) if c.chunk_type == "module_docstring"]
        assert len(docs) == 1
        assert "Module for solving Poisson problems." in docs[0].text

    def test_not_emitted_when_absent(self, tmp_path):
        src = """\
            def solve(x, y, z):
                a = x + y
                b = a * z
                return b
        """
        p = write(tmp_path, "mod.py", src)
        docs = [c for c in chunk_file(p) if c.chunk_type == "module_docstring"]
        assert len(docs) == 0


# ── chunk type invariant ──────────────────────────────────────────────────────

class TestChunkTypeInvariant:
    def test_all_emitted_types_are_valid(self, tmp_path):
        src = """\
            \"\"\"Module doc.\"\"\"

            class Solver:
                \"\"\"Solver class.\"\"\"

                def solve(self, x, y, z):
                    a = x + y
                    b = a * z
                    return b

            def helper(x, y, z):
                a = x * y
                b = a - z
                return b
        """
        p = write(tmp_path, "mod.py", src)
        for chunk in chunk_file(p):
            assert chunk.chunk_type in VALID_CHUNK_TYPES