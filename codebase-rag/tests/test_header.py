"""Tests for the context header builder (Step 5).

Interface:
    build_header(chunk: CoarseChunk, source: str, all_chunks: list[CoarseChunk]) -> str

Output format — '#'-prefixed lines, sections omitted when empty, fixed order:
    # class: ClassName                  (only when chunk.parent_class is not None)
    # import numpy as np                (full import line from source, one per referenced import)
    # above: def prev(args):            (first def/async def line of nearest preceding chunk)
    # below: def next(args):            (first def/async def line of nearest following chunk)

Neighbor rules:
    - Filter all_chunks to chunk_type in {"function", "method"}, sort by line_range[0].
    - above = nearest chunk whose start line < chunk's start line.
    - below = nearest chunk whose start line > chunk's start line.
    - class_header and module_docstring chunks are never neighbors.

Import filtering rules:
    - Parse source line-by-line for lines starting with "import" or "from".
    - Extract usable name(s): alias if "as" present, else module/attribute name.
      "import numpy as np"          -> usable: "np"
      "import numpy"                -> usable: "numpy"
      "from scipy import sparse"    -> usable: "sparse"
      "from scipy import a, b"      -> usable: "a", "b"
    - Include a canonical single-line form of the import if any usable name appears 
    as a whole word in chunk.text.

Neighbor signature:
    - First line of neighbor.text that starts with "def" or "async def" (stripped).

Returns "" (empty string, not None) when no sections present.
All non-empty output lines start with "#".
"""
from kernelpack_rag.chunking.coarse import CoarseChunk
from kernelpack_rag.chunking.header import build_header


# ── fixture helper ────────────────────────────────────────────────────────────

def make_chunk(
    text: str,
    qualname: str,
    chunk_type: str,
    line_range: tuple[int, int],
    source_file: str = "mod.py",
    parent_class: str | None = None,
) -> CoarseChunk:
    return CoarseChunk(
        text=text,
        qualname=qualname,
        chunk_type=chunk_type,
        line_range=line_range,
        source_file=source_file,
        parent_class=parent_class,
        module="kernelpack.test"
    )


# ── class line ────────────────────────────────────────────────────────────────

class TestClassLine:
    def test_method_has_class_line(self):
        chunk = make_chunk(
            text="def solve(self, rhs, tol):\n    result = self._compute(rhs)\n    return result / tol",
            qualname="PoissonSolver.solve",
            chunk_type="method",
            line_range=(5, 7),
            parent_class="PoissonSolver",
        )
        header = build_header(chunk, source="", all_chunks=[chunk])
        assert "# class: PoissonSolver" in header

    def test_standalone_function_no_class_line(self):
        chunk = make_chunk(
            text="def helper(x, y):\n    return x + y",
            qualname="helper",
            chunk_type="function",
            line_range=(1, 2),
        )
        header = build_header(chunk, source="", all_chunks=[chunk])
        assert not any("class:" in line for line in header.splitlines())

    def test_class_header_chunk_no_class_line(self):
        # class_header chunks have parent_class=None — no class line
        chunk = make_chunk(
            text="class Solver:\n    pass",
            qualname="Solver",
            chunk_type="class_header",
            line_range=(1, 2),
        )
        header = build_header(chunk, source="", all_chunks=[chunk])
        assert not any("class:" in line for line in header.splitlines())


# ── import filtering ──────────────────────────────────────────────────────────

class TestImportFiltering:
    SOURCE = (
        "import numpy as np\n"
        "from scipy import sparse\n"
        "import os\n"
        "\n"
        "def solve(x):\n"
        "    result = np.zeros(x)\n"
        "    return sparse.linalg.spsolve(result, x)\n"
    )

    def test_referenced_import_included(self):
        chunk = make_chunk(
            text="def solve(x):\n    result = np.zeros(x)\n    return result",
            qualname="solve",
            chunk_type="function",
            line_range=(5, 7),
        )
        header = build_header(chunk, self.SOURCE, all_chunks=[chunk])
        assert any("numpy" in line for line in header.splitlines())

    def test_unreferenced_import_excluded(self):
        # chunk uses np and sparse but not os
        chunk = make_chunk(
            text="def solve(x):\n    result = np.zeros(x)\n    return sparse.linalg.spsolve(result, x)",
            qualname="solve",
            chunk_type="function",
            line_range=(5, 7),
        )
        header = build_header(chunk, self.SOURCE, all_chunks=[chunk])
        assert not any("import os" in line for line in header.splitlines())

    def test_alias_matched(self):
        # "import numpy as np" — chunk uses "np", not "numpy"
        source = "import numpy as np\n\ndef f(x):\n    return np.zeros(x)\n"
        chunk = make_chunk(
            text="def f(x):\n    return np.zeros(x)",
            qualname="f",
            chunk_type="function",
            line_range=(3, 4),
        )
        header = build_header(chunk, source, all_chunks=[chunk])
        assert any("numpy" in line for line in header.splitlines())

    def test_no_imports_in_source(self):
        source = "def f(x):\n    return x\n"
        chunk = make_chunk(
            text="def f(x):\n    return x",
            qualname="f",
            chunk_type="function",
            line_range=(1, 2),
        )
        header = build_header(chunk, source, all_chunks=[chunk])
        assert not any("import" in line for line in header.splitlines())

    def test_from_import_included_when_referenced(self):
        source = "from scipy import sparse\n\ndef f(x):\n    return sparse.eye(x)\n"
        chunk = make_chunk(
            text="def f(x):\n    return sparse.eye(x)",
            qualname="f",
            chunk_type="function",
            line_range=(3, 4),
        )
        header = build_header(chunk, source, all_chunks=[chunk])
        assert any("scipy" in line for line in header.splitlines())

    def test_multi_line_import_included(self):
        source = (
            "from scipy import (\n"
            "    sparse,\n"
            "    linalg,\n"
            ")\n"
            "\n"
            "def f(x):\n"
            "    return sparse.eye(x)\n"
        )
        chunk = make_chunk(
            text="def f(x):\n    return sparse.eye(x)",
            qualname="f",
            chunk_type="function",
            line_range=(6, 7),
        )
        header = build_header(chunk, source, all_chunks=[chunk])
        assert any("scipy" in line for line in header.splitlines())

    def test_multi_line_import_excluded_when_unreferenced(self):
        source = (
            "from scipy import (\n"
            "    sparse,\n"
            "    linalg,\n"
            ")\n"
            "\n"
            "def f(x):\n"
            "    return x\n"
        )
        chunk = make_chunk(
            text="def f(x):\n    return x",
            qualname="f",
            chunk_type="function",
            line_range=(6, 7),
        )
        header = build_header(chunk, source, all_chunks=[chunk])
        assert not any("scipy" in line for line in header.splitlines())


# ── neighbors ─────────────────────────────────────────────────────────────────

class TestNeighbors:
    def _three_chunks(self):
        above = make_chunk(
            text="def setup(self, domain):\n    self.domain = domain\n    self.initialized = True\n    return self",
            qualname="Solver.setup",
            chunk_type="method",
            line_range=(3, 6),
            parent_class="Solver",
        )
        target = make_chunk(
            text="def solve(self, rhs, tol, max_iter):\n    result = self._compute(rhs)\n    scaled = result * tol\n    return scaled / max_iter",
            qualname="Solver.solve",
            chunk_type="method",
            line_range=(8, 11),
            parent_class="Solver",
        )
        below = make_chunk(
            text="def cleanup(self):\n    self.domain = None\n    self.result = None\n    return True",
            qualname="Solver.cleanup",
            chunk_type="method",
            line_range=(13, 16),
            parent_class="Solver",
        )
        return above, target, below

    def test_above_neighbor_present(self):
        above, target, below = self._three_chunks()
        header = build_header(target, source="", all_chunks=[above, target, below])
        assert any("above:" in line for line in header.splitlines())

    def test_below_neighbor_present(self):
        above, target, below = self._three_chunks()
        header = build_header(target, source="", all_chunks=[above, target, below])
        assert any("below:" in line for line in header.splitlines())

    def test_above_contains_correct_name(self):
        above, target, below = self._three_chunks()
        header = build_header(target, source="", all_chunks=[above, target, below])
        above_lines = [l for l in header.splitlines() if "above:" in l]
        assert len(above_lines) == 1
        assert "setup" in above_lines[0]

    def test_below_contains_correct_name(self):
        above, target, below = self._three_chunks()
        header = build_header(target, source="", all_chunks=[above, target, below])
        below_lines = [l for l in header.splitlines() if "below:" in l]
        assert len(below_lines) == 1
        assert "cleanup" in below_lines[0]

    def test_above_is_signature_only(self):
        above, target, below = self._three_chunks()
        header = build_header(target, source="", all_chunks=[above, target, below])
        above_lines = [l for l in header.splitlines() if "above:" in l]
        assert "self.domain" not in above_lines[0]

    def test_below_is_signature_only(self):
        above, target, below = self._three_chunks()
        header = build_header(target, source="", all_chunks=[above, target, below])
        below_lines = [l for l in header.splitlines() if "below:" in l]
        assert "self.domain = None" not in below_lines[0]

    def test_no_above_when_first(self):
        above, target, below = self._three_chunks()
        header = build_header(above, source="", all_chunks=[above, target, below])
        assert not any("above:" in line for line in header.splitlines())

    def test_no_below_when_last(self):
        above, target, below = self._three_chunks()
        header = build_header(below, source="", all_chunks=[above, target, below])
        assert not any("below:" in line for line in header.splitlines())

    def test_class_header_not_a_neighbor(self):
        # class_header chunks should never appear as above/below
        class_hdr = make_chunk(
            text="class Solver:\n    pass",
            qualname="Solver",
            chunk_type="class_header",
            line_range=(1, 2),
        )
        target = make_chunk(
            text="def solve(self, rhs, tol, max_iter):\n    result = self._compute(rhs)\n    scaled = result * tol\n    return scaled",
            qualname="Solver.solve",
            chunk_type="method",
            line_range=(4, 7),
            parent_class="Solver",
        )
        header = build_header(target, source="", all_chunks=[class_hdr, target])
        assert not any("above:" in line for line in header.splitlines())

    def test_module_docstring_not_a_neighbor(self):
        doc = make_chunk(
            text='"""Module docstring."""',
            qualname="mod",
            chunk_type="module_docstring",
            line_range=(1, 1),
        )
        target = make_chunk(
            text="def solve(x, y, z, w):\n    return x + y + z + w",
            qualname="solve",
            chunk_type="function",
            line_range=(3, 4),
        )
        header = build_header(target, source="", all_chunks=[doc, target])
        assert not any("above:" in line for line in header.splitlines())


# ── format invariants ─────────────────────────────────────────────────────────

class TestFormatInvariants:
    def test_return_type_is_str(self):
        chunk = make_chunk(
            text="def f(x):\n    return x",
            qualname="f",
            chunk_type="function",
            line_range=(1, 2),
        )
        assert isinstance(build_header(chunk, source="", all_chunks=[chunk]), str)

    def test_empty_when_no_context(self):
        # no parent class, no imports, no neighbors
        chunk = make_chunk(
            text="def f(x):\n    return x",
            qualname="f",
            chunk_type="function",
            line_range=(1, 2),
        )
        assert build_header(chunk, source="", all_chunks=[chunk]) == ""

    def test_all_lines_hash_prefixed(self):
        source = "import numpy as np\n\ndef setup(x):\n    return x\n\ndef solve(x):\n    return np.zeros(x)\n"
        above = make_chunk(
            text="def setup(x):\n    return x",
            qualname="setup",
            chunk_type="function",
            line_range=(3, 4),
        )
        chunk = make_chunk(
            text="def solve(x):\n    return np.zeros(x)",
            qualname="solve",
            chunk_type="function",
            line_range=(6, 7),
        )
        header = build_header(chunk, source, all_chunks=[above, chunk])
        for line in header.splitlines():
            assert line.startswith("#"), f"Line not '#'-prefixed: {line!r}"

    def test_section_order(self):
        # class → imports → above → below
        source = "import numpy as np\n"
        above = make_chunk(
            text="def setup(self, x, y):\n    self.x = x\n    return self",
            qualname="Solver.setup",
            chunk_type="method",
            line_range=(3, 5),
            parent_class="Solver",
        )
        chunk = make_chunk(
            text="def solve(self, rhs):\n    return np.zeros(rhs)",
            qualname="Solver.solve",
            chunk_type="method",
            line_range=(7, 8),
            parent_class="Solver",
        )
        below = make_chunk(
            text="def cleanup(self, x, y):\n    self.x = None\n    return True",
            qualname="Solver.cleanup",
            chunk_type="method",
            line_range=(10, 12),
            parent_class="Solver",
        )
        header = build_header(chunk, source, all_chunks=[above, chunk, below])
        lines = header.splitlines()
        positions = {}
        for i, line in enumerate(lines):
            if "class:" in line:
                positions["class"] = i
            elif "import" in line:
                positions.setdefault("import", i)
            elif "above:" in line:
                positions["above"] = i
            elif "below:" in line:
                positions["below"] = i

        if "class" in positions and "import" in positions:
            assert positions["class"] < positions["import"]
        if "import" in positions and "above" in positions:
            assert positions["import"] < positions["above"]
        if "above" in positions and "below" in positions:
            assert positions["above"] < positions["below"]