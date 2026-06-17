"""
TDD tests for chunking/metadata.py (Step 7 of the RAG pipeline).

Contract being specified:

    build_symbol_table(package_root: Path) -> dict[str, uuid.UUID]
        Walk src/kernelpack/, collect every qualified name of the form
            module.ClassName
            module.ClassName.method_name
            module.function_name
        (where "module" is the dot-separated path relative to src/kernelpack,
        e.g. "geometry.core", "_numba", "solvers.diffusion")
        and map each to a deterministic UUID:
            uuid.uuid5(KP_NAMESPACE, f"code:{source_file}:{qualname}:coarse:0")
        Dunder methods (__init__, __post_init__, …) are excluded.
        source_file is the absolute path to the .py file.

    extract_metadata(chunk: CoarseChunk, symbol_table: dict) -> ChunkMetadata
        Fields on the returned dataclass:
            module          str
            function_name   str   (leaf name only, no class prefix)
            source_file     str
            line_range      tuple[int, int]
            chunk_type      str   "function" | "method" | "class"
            parent_class    str | None
            math_terms      list[str]   subset of data/math_terms.json lexicon
            cross_refs      list[str]   intra-package qualnames found in chunk AST
            cross_ref_ids   list[uuid.UUID]
            has_numba       bool

CoarseChunk proxy (real class lives in chunking/types.py when implemented):
    source_file: str
    module:      str
    qualname:    str
    ast_node:    ast.AST
    line_range:  tuple[int, int]
"""

from __future__ import annotations

import ast
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from kernelpack_rag.chunking.metadata import ChunkMetadata
from kernelpack_rag.chunking.metadata import build_symbol_table
from kernelpack_rag.chunking.metadata import extract_metadata

# ── constants ──────────────────────────────────────────────────────────────────

KP_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

REPO_ROOT = Path(__file__).parent.parent          
KP_ROOT = Path("/Users/jordanchambers/public-projects/kernelpack-python")
SRC_ROOT = KP_ROOT / "src" / "kernelpack"

def _rel(path: Path) -> str:
    return str(path.relative_to(KP_ROOT))         

DATA_DIR = REPO_ROOT / "kernelpack_rag" / "data"                    

# ── local CoarseChunk proxy ────────────────────────────────────────────────────


@dataclass(frozen=True)
class CoarseChunk:
    """Minimal stand-in for the real CoarseChunk from chunking/types.py."""

    source_file: str
    module: str
    qualname: str
    ast_node: ast.AST
    line_range: tuple


# ── helpers ────────────────────────────────────────────────────────────────────


def _expected_uuid(source_file: str, qualname: str) -> uuid.UUID:
    return uuid.uuid5(KP_NAMESPACE, f"code:{source_file}:{qualname}:coarse:0")


def _parse(rel: str) -> ast.Module:
    return ast.parse((KP_ROOT / rel).read_text())


def _find_func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise LookupError(name)


def _find_method(tree: ast.Module, cls: str, method: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for child in ast.walk(node):
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == method
                ):
                    return child
    raise LookupError(f"{cls}.{method}")


def _make_chunk(src: str, module: str, qualname: str, node: ast.AST) -> CoarseChunk:
    lo = getattr(node, "lineno", 0)
    hi = getattr(node, "end_lineno", lo)
    return CoarseChunk(source_file=src, module=module, qualname=qualname, ast_node=node, line_range=(lo, hi))


# ── shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def symbol_table() -> dict:
    return build_symbol_table(SRC_ROOT)


@pytest.fixture(scope="session")
def geo_tree() -> ast.Module:
    return _parse("src/kernelpack/geometry/core.py")


@pytest.fixture(scope="session")
def rbffd_tree() -> ast.Module:
    return _parse("src/kernelpack/rbffd/core.py")


@pytest.fixture(scope="session")
def numba_tree() -> ast.Module:
    return _parse("src/kernelpack/_numba.py")


@pytest.fixture(scope="session")
def diffusion_tree() -> ast.Module:
    return _parse("src/kernelpack/solvers/diffusion.py")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  build_symbol_table
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildSymbolTable:

    # ── basic structure ──────────────────────────────────────────────────────

    def test_returns_dict(self, symbol_table):
        assert isinstance(symbol_table, dict)

    def test_keys_are_strings(self, symbol_table):
        sample = list(symbol_table.keys())[:30]
        assert all(isinstance(k, str) for k in sample)

    def test_values_are_uuids(self, symbol_table):
        sample = list(symbol_table.values())[:30]
        assert all(isinstance(v, uuid.UUID) for v in sample)

    def test_non_empty(self, symbol_table):
        assert len(symbol_table) > 50

    # ── required keys: geometry/core.py ─────────────────────────────────────

    def test_geometry_top_level_function_phs_kernel(self, symbol_table):
        assert "kernelpack.geometry.core.phs_kernel" in symbol_table

    def test_geometry_top_level_function_distance_matrix(self, symbol_table):
        assert "kernelpack.geometry.core.distance_matrix" in symbol_table

    def test_geometry_top_level_function_normalize_rows(self, symbol_table):
        assert "kernelpack.geometry.core.normalize_rows" in symbol_table

    def test_geometry_class_rbflevelset(self, symbol_table):
        assert "kernelpack.geometry.core.RBFLevelSet" in symbol_table

    def test_geometry_method_rbflevelset_evaluate(self, symbol_table):
        assert "kernelpack.geometry.core.RBFLevelSet.evaluate" in symbol_table

    def test_geometry_method_rbflevelset_build_level_set_from_cfi(self, symbol_table):
        assert "kernelpack.geometry.core.RBFLevelSet.build_level_set_from_cfi" in symbol_table

    def test_geometry_class_embeddedsurface(self, symbol_table):
        assert "kernelpack.geometry.core.EmbeddedSurface" in symbol_table

    def test_geometry_method_embeddedsurface_build_level_set_from_geometric_model(self, symbol_table):
        assert "kernelpack.geometry.core.EmbeddedSurface.build_level_set_from_geometric_model" in symbol_table

    def test_geometry_method_embeddedsurface_build_closed_geometric_model_ps(self, symbol_table):
        assert "kernelpack.geometry.core.EmbeddedSurface.build_closed_geometric_model_ps" in symbol_table

    # ── required keys: rbffd/core.py ────────────────────────────────────────

    def test_rbffd_class_stencil_properties(self, symbol_table):
        assert "kernelpack.rbffd.core.StencilProperties" in symbol_table

    def test_rbffd_class_rbfstencil(self, symbol_table):
        assert "kernelpack.rbffd.core.RBFStencil" in symbol_table

    def test_rbffd_method_rbfstencil_lap_op(self, symbol_table):
        assert "kernelpack.rbffd.core.RBFStencil.lap_op" in symbol_table

    def test_rbffd_method_rbfstencil_initialize_geometry(self, symbol_table):
        assert "kernelpack.rbffd.core.RBFStencil.initialize_geometry" in symbol_table

    def test_rbffd_method_rbfstencil_compute_weights(self, symbol_table):
        assert "kernelpack.rbffd.core.RBFStencil.compute_weights" in symbol_table

    def test_rbffd_class_fddiffop(self, symbol_table):
        assert "kernelpack.rbffd.core.FDDiffOp" in symbol_table

    def test_rbffd_method_fddiffop_assemble_op(self, symbol_table):
        assert "kernelpack.rbffd.core.FDDiffOp.assemble_op" in symbol_table

    def test_rbffd_method_fddiffop_get_op(self, symbol_table):
        assert "kernelpack.rbffd.core.FDDiffOp.get_op" in symbol_table

    def test_rbffd_class_wlsstencil(self, symbol_table):
        assert "kernelpack.rbffd.core.WeightedLeastSquaresStencil" in symbol_table

    # ── required keys: domain, nodes, poly ──────────────────────────────────

    def test_domain_class_domain_descriptor(self, symbol_table):
        assert "kernelpack.domain.core.DomainDescriptor" in symbol_table

    def test_domain_method_build_structs(self, symbol_table):
        assert "kernelpack.domain.core.DomainDescriptor.build_structs" in symbol_table

    def test_domain_dual_class(self, symbol_table):
        assert "kernelpack.domain.dual.DualNodeDomainDescriptor" in symbol_table

    def test_nodes_class_domain_node_generator(self, symbol_table):
        assert "kernelpack.nodes.core.DomainNodeGenerator" in symbol_table

    def test_poly_class_polynomial_basis(self, symbol_table):
        assert "kernelpack.poly.core.PolynomialBasis" in symbol_table

    def test_poly_method_evaluate(self, symbol_table):
        assert "kernelpack.poly.core.PolynomialBasis.evaluate" in symbol_table

    def test_poly_method_from_total_degree(self, symbol_table):
        assert "kernelpack.poly.core.PolynomialBasis.from_total_degree" in symbol_table

    # ── required keys: solvers ───────────────────────────────────────────────

    def test_solver_poisson(self, symbol_table):
        assert "kernelpack.solvers.poisson.PoissonSolver" in symbol_table

    def test_solver_diffusion(self, symbol_table):
        assert "kernelpack.solvers.diffusion.DiffusionSolver" in symbol_table

    def test_solver_diffusion_method_bdf1_step(self, symbol_table):
        assert "kernelpack.solvers.diffusion.DiffusionSolver.bdf1_step" in symbol_table

    # ── required keys: _numba (public wrappers) ──────────────────────────────

    def test_numba_dense_distance_matrix(self, symbol_table):
        assert "kernelpack._numba.dense_distance_matrix" in symbol_table

    def test_numba_phs_kernel_matrix(self, symbol_table):
        assert "kernelpack._numba.phs_kernel_matrix" in symbol_table

    def test_numba_build_augmented_rbf_lhs(self, symbol_table):
        assert "kernelpack._numba.build_augmented_rbf_lhs" in symbol_table

    # ── dunder exclusion ─────────────────────────────────────────────────────

    def test_no_dunder_init(self, symbol_table):
        for key in symbol_table:
            assert not key.endswith(".__init__"), f"dunder leaked: {key}"

    def test_no_dunder_post_init(self, symbol_table):
        for key in symbol_table:
            assert not key.endswith(".__post_init__"), f"dunder leaked: {key}"

    def test_no_dunder_repr(self, symbol_table):
        for key in symbol_table:
            assert not key.endswith(".__repr__"), f"dunder leaked: {key}"

    # ── UUID formula ─────────────────────────────────────────────────────────

    def test_uuid_formula_phs_kernel(self, symbol_table):
        qualname = "kernelpack.geometry.core.phs_kernel"
        src = _rel(SRC_ROOT / "geometry" / "core.py")
        assert symbol_table[qualname] == _expected_uuid(src, qualname)

    def test_uuid_formula_rbfstencil_lap_op(self, symbol_table):
        qualname = "kernelpack.rbffd.core.RBFStencil.lap_op"
        src = _rel(SRC_ROOT / "rbffd" / "core.py")
        assert symbol_table[qualname] == _expected_uuid(src, qualname)

    def test_uuid_formula_fddiffop_assemble_op(self, symbol_table):
        qualname = "kernelpack.rbffd.core.FDDiffOp.assemble_op"
        src = _rel(SRC_ROOT / "rbffd" / "core.py")
        assert symbol_table[qualname] == _expected_uuid(src, qualname)

    def test_uuid_formula_numba_dense_distance_matrix(self, symbol_table):
        qualname = "kernelpack._numba.dense_distance_matrix"
        src = _rel(SRC_ROOT / "_numba.py")
        assert symbol_table[qualname] == _expected_uuid(src, qualname)

    def test_uuids_distinct_for_different_names(self, symbol_table):
        id1 = symbol_table["kernelpack.geometry.core.phs_kernel"]
        id2 = symbol_table["kernelpack.geometry.core.distance_matrix"]
        assert id1 != id2

    def test_uuids_distinct_across_modules(self, symbol_table):
        id1 = symbol_table["kernelpack.geometry.core.RBFLevelSet"]
        id2 = symbol_table["kernelpack.rbffd.core.RBFStencil"]
        assert id1 != id2

    def test_stable_across_calls(self):
        t1 = build_symbol_table(SRC_ROOT)
        t2 = build_symbol_table(SRC_ROOT)
        assert t1 == t2


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ChunkMetadata field types  (using phs_kernel as a simple reference chunk)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def phs_kernel_meta(symbol_table, geo_tree):
    node = _find_func(geo_tree, "phs_kernel")
    src = str(SRC_ROOT / "geometry" / "core.py")
    chunk = _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node)
    return extract_metadata(chunk, symbol_table)


class TestChunkMetadataFieldTypes:

    def test_returns_chunk_metadata_instance(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta, ChunkMetadata)

    def test_module_is_str(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta.module, str)

    def test_function_name_is_str(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta.function_name, str)

    def test_source_file_is_str(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta.source_file, str)

    def test_line_range_is_two_int_tuple(self, phs_kernel_meta):
        lo, hi = phs_kernel_meta.line_range
        assert isinstance(lo, int) and isinstance(hi, int)

    def test_chunk_type_is_str(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta.chunk_type, str)

    def test_parent_class_is_none_or_str(self, phs_kernel_meta):
        assert phs_kernel_meta.parent_class is None or isinstance(phs_kernel_meta.parent_class, str)

    def test_math_terms_is_list(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta.math_terms, list)

    def test_cross_refs_is_list(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta.cross_refs, list)

    def test_cross_ref_ids_is_list_of_uuids(self, phs_kernel_meta):
        assert all(isinstance(x, uuid.UUID) for x in phs_kernel_meta.cross_ref_ids)

    def test_has_numba_is_bool(self, phs_kernel_meta):
        assert isinstance(phs_kernel_meta.has_numba, bool)

    def test_cross_refs_and_ids_same_length(self, phs_kernel_meta):
        assert len(phs_kernel_meta.cross_refs) == len(phs_kernel_meta.cross_ref_ids)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  module / function_name / source_file propagation
# ══════════════════════════════════════════════════════════════════════════════


class TestModuleAndName:

    def test_top_level_function_name(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert meta.function_name == "phs_kernel"

    def test_module_propagated(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert meta.module == "geometry.core"

    def test_source_file_propagated(self, symbol_table, geo_tree):
        src = str(SRC_ROOT / "geometry" / "core.py")
        node = _find_func(geo_tree, "phs_kernel")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert meta.source_file == src

    def test_method_function_name_leaf_only(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "RBFLevelSet", "evaluate")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.RBFLevelSet.evaluate", method),
            symbol_table,
        )
        assert meta.function_name == "evaluate"

    def test_different_module_propagated(self, symbol_table, rbffd_tree):
        method = _find_method(rbffd_tree, "FDDiffOp", "assemble_op")
        src = str(SRC_ROOT / "rbffd" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "rbffd.core", "rbffd.core.FDDiffOp.assemble_op", method),
            symbol_table,
        )
        assert meta.module == "rbffd.core"
        assert meta.function_name == "assemble_op"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  chunk_type  and  parent_class
# ══════════════════════════════════════════════════════════════════════════════


class TestChunkTypeAndParentClass:

    def test_top_level_function_chunk_type(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert meta.chunk_type == "function"

    def test_top_level_function_no_parent(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert meta.parent_class is None

    def test_method_chunk_type(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "RBFLevelSet", "evaluate")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.RBFLevelSet.evaluate", method),
            symbol_table,
        )
        assert meta.chunk_type == "method"

    def test_method_parent_class_correct(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "RBFLevelSet", "evaluate")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.RBFLevelSet.evaluate", method),
            symbol_table,
        )
        assert meta.parent_class == "RBFLevelSet"

    def test_embedded_surface_method_parent(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "EmbeddedSurface", "build_level_set_from_geometric_model")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.EmbeddedSurface.build_level_set_from_geometric_model"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        assert meta.parent_class == "EmbeddedSurface"
        assert meta.chunk_type == "method"

    def test_fddiffop_assemble_op_parent(self, symbol_table, rbffd_tree):
        method = _find_method(rbffd_tree, "FDDiffOp", "assemble_op")
        src = str(SRC_ROOT / "rbffd" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "rbffd.core", "rbffd.core.FDDiffOp.assemble_op", method),
            symbol_table,
        )
        assert meta.parent_class == "FDDiffOp"
        assert meta.chunk_type == "method"

    def test_normalize_rows_no_parent(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "normalize_rows")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.normalize_rows", node),
            symbol_table,
        )
        assert meta.parent_class is None
        assert meta.chunk_type == "function"


# ══════════════════════════════════════════════════════════════════════════════
# 5.  line_range
# ══════════════════════════════════════════════════════════════════════════════


class TestLineRange:

    def test_line_range_matches_ast_node(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert meta.line_range == (node.lineno, node.end_lineno)

    def test_line_range_positive(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "distance_matrix")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.distance_matrix", node),
            symbol_table,
        )
        lo, hi = meta.line_range
        assert lo >= 1 and hi >= lo

    def test_line_range_method(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "RBFLevelSet", "build_level_set_from_cfi")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.RBFLevelSet.build_level_set_from_cfi"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        assert meta.line_range == (method.lineno, method.end_lineno)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  has_numba
# ══════════════════════════════════════════════════════════════════════════════


class TestHasNumba:

    def test_plain_function_false(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert meta.has_numba is False

    def test_plain_method_false(self, symbol_table, rbffd_tree):
        method = _find_method(rbffd_tree, "FDDiffOp", "assemble_op")
        src = str(SRC_ROOT / "rbffd" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "rbffd.core", "rbffd.core.FDDiffOp.assemble_op", method),
            symbol_table,
        )
        assert meta.has_numba is False

    def test_njit_decorated_function_true(self, symbol_table, numba_tree):
        # _distance_matrix_numba is decorated @njit(cache=True, fastmath=True)
        node = _find_func(numba_tree, "_distance_matrix_numba")
        src = str(SRC_ROOT / "_numba.py")
        meta = extract_metadata(
            _make_chunk(src, "_numba", "_numba._distance_matrix_numba", node),
            symbol_table,
        )
        assert meta.has_numba is True

    def test_second_njit_function_true(self, symbol_table, numba_tree):
        # _phs_kernel_numba is also @njit decorated
        node = _find_func(numba_tree, "_phs_kernel_numba")
        src = str(SRC_ROOT / "_numba.py")
        meta = extract_metadata(
            _make_chunk(src, "_numba", "_numba._phs_kernel_numba", node),
            symbol_table,
        )
        assert meta.has_numba is True

    def test_numba_wrapper_without_decorator_false(self, symbol_table, numba_tree):
        # dense_distance_matrix is a plain public wrapper — no @njit on it
        node = _find_func(numba_tree, "dense_distance_matrix")
        src = str(SRC_ROOT / "_numba.py")
        meta = extract_metadata(
            _make_chunk(src, "_numba", "_numba.dense_distance_matrix", node),
            symbol_table,
        )
        assert meta.has_numba is False

    def test_normalize_rows_false(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "normalize_rows")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.normalize_rows", node),
            symbol_table,
        )
        assert meta.has_numba is False

    def test_phs_lap_numba_true(self, symbol_table, numba_tree):
        node = _find_func(numba_tree, "_phs_lap_numba")
        src = str(SRC_ROOT / "_numba.py")
        meta = extract_metadata(
            _make_chunk(src, "_numba", "_numba._phs_lap_numba", node),
            symbol_table,
        )
        assert meta.has_numba is True


# ══════════════════════════════════════════════════════════════════════════════
# 7.  math_terms
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def lexicon() -> set:
    return set(json.loads((DATA_DIR / "math_terms.json").read_text()))


class TestMathTerms:

    def test_math_terms_is_list(self, symbol_table, geo_tree, lexicon):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert isinstance(meta.math_terms, list)

    def test_math_terms_subset_of_lexicon(self, symbol_table, geo_tree, lexicon):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert set(meta.math_terms).issubset(lexicon), (
            f"terms not in lexicon: {set(meta.math_terms) - lexicon}"
        )

    def test_no_duplicate_math_terms(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "distance_matrix")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.distance_matrix", node),
            symbol_table,
        )
        assert len(meta.math_terms) == len(set(meta.math_terms))

    def test_phs_kernel_has_phs_term(self, symbol_table, geo_tree, lexicon):
        # "phs" is in lexicon AND appears as a token in the identifier "phs_kernel"
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert "phs" in meta.math_terms

    def test_rbflevelset_build_cfi_has_rbf_term(self, symbol_table, geo_tree, lexicon):
        # build_level_set_from_cfi's docstring and identifiers contain "rbf"
        method = _find_method(geo_tree, "RBFLevelSet", "build_level_set_from_cfi")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.RBFLevelSet.build_level_set_from_cfi"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        # "rbf" appears from "RBF" in class name token and from "phs_kernel" body call
        assert "rbf" in meta.math_terms or "phs" in meta.math_terms

    def test_laplacian_term_in_phs_lap_numba(self, symbol_table, numba_tree, lexicon):
        # _phs_lap_numba — the identifier "lap" tokenises; "laplacian" is in lexicon.
        # Matching strategy may map "lap" → "laplacian" OR the docstring mentions it.
        node = _find_func(numba_tree, "_phs_lap_numba")
        src = str(SRC_ROOT / "_numba.py")
        meta = extract_metadata(
            _make_chunk(src, "_numba", "_numba._phs_lap_numba", node),
            symbol_table,
        )
        assert "laplacian" in meta.math_terms or "phs" in meta.math_terms

    def test_stencil_term_in_fddiffop_assemble_op(self, symbol_table, rbffd_tree, lexicon):
        method = _find_method(rbffd_tree, "FDDiffOp", "assemble_op")
        src = str(SRC_ROOT / "rbffd" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "rbffd.core", "rbffd.core.FDDiffOp.assemble_op", method),
            symbol_table,
        )
        assert "stencil" in meta.math_terms or "stencil assembly" in meta.math_terms

    def test_math_terms_all_lowercase(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        assert all(t == t.lower() for t in meta.math_terms)

    def test_no_python_builtins_in_math_terms(self, symbol_table, geo_tree, lexicon):
        node = _find_func(geo_tree, "normalize_rows")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.normalize_rows", node),
            symbol_table,
        )
        bad = {"self", "return", "import", "def", "class", "none", "true", "false"}
        assert not bad.intersection(set(meta.math_terms))

    def test_method_math_terms_subset_of_lexicon(self, symbol_table, rbffd_tree, lexicon):
        method = _find_method(rbffd_tree, "RBFStencil", "lap_op")
        src = str(SRC_ROOT / "rbffd" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "rbffd.core", "rbffd.core.RBFStencil.lap_op", method),
            symbol_table,
        )
        assert set(meta.math_terms).issubset(lexicon)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  cross_refs  and  cross_ref_ids
# ══════════════════════════════════════════════════════════════════════════════


class TestCrossRefs:

    def test_cross_refs_all_in_symbol_table(self, symbol_table, geo_tree):
        # phs_kernel is a 1-line wrapper; any cross_refs found must be in the table
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        for ref in meta.cross_refs:
            assert ref in symbol_table, f"{ref!r} not in symbol_table"

    def test_cross_ref_ids_match_symbol_table(self, symbol_table, geo_tree):
        node = _find_func(geo_tree, "phs_kernel")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.phs_kernel", node),
            symbol_table,
        )
        for ref, uid in zip(meta.cross_refs, meta.cross_ref_ids):
            assert symbol_table[ref] == uid

    def test_normalize_rows_has_no_intra_package_calls(self, symbol_table, geo_tree):
        # normalize_rows calls only numpy — no intra-package cross_refs
        node = _find_func(geo_tree, "normalize_rows")
        src = str(SRC_ROOT / "geometry" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", "geometry.core.normalize_rows", node),
            symbol_table,
        )
        assert meta.cross_refs == []
        assert meta.cross_ref_ids == []

    def test_build_level_set_from_cfi_cross_refs_intra_module(self, symbol_table, geo_tree):
        # build_level_set_from_cfi calls normalize_rows, distance_matrix, phs_kernel
        # — all public functions in the same module.
        method = _find_method(geo_tree, "RBFLevelSet", "build_level_set_from_cfi")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.RBFLevelSet.build_level_set_from_cfi"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        assert "kernelpack.geometry.core.normalize_rows" in meta.cross_refs
        assert "kernelpack.geometry.core.distance_matrix" in meta.cross_refs
        assert "kernelpack.geometry.core.phs_kernel" in meta.cross_refs

    def test_build_level_set_from_cfi_cross_ref_ids_consistent(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "RBFLevelSet", "build_level_set_from_cfi")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.RBFLevelSet.build_level_set_from_cfi"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        for ref, uid in zip(meta.cross_refs, meta.cross_ref_ids):
            assert symbol_table[ref] == uid

    def test_build_level_set_from_geometric_model_refs_rbflevelset(self, symbol_table, geo_tree):
        # build_level_set_from_geometric_model constructs RBFLevelSet() directly
        method = _find_method(geo_tree, "EmbeddedSurface", "build_level_set_from_geometric_model")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.EmbeddedSurface.build_level_set_from_geometric_model"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        assert "kernelpack.geometry.core.RBFLevelSet" in meta.cross_refs

    def test_cross_refs_no_external_packages(self, symbol_table, rbffd_tree):
        # assemble_op references private helpers in the same module;
        # numpy / scipy names must not appear in cross_refs.
        method = _find_method(rbffd_tree, "FDDiffOp", "assemble_op")
        src = str(SRC_ROOT / "rbffd" / "core.py")
        meta = extract_metadata(
            _make_chunk(src, "rbffd.core", "rbffd.core.FDDiffOp.assemble_op", method),
            symbol_table,
        )
        for ref in meta.cross_refs:
            assert not ref.startswith("np."), f"external numpy ref leaked: {ref}"
            assert not ref.startswith("scipy."), f"external scipy ref leaked: {ref}"

    def test_cross_refs_no_stdlib_names(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "RBFLevelSet", "build_level_set_from_cfi")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.RBFLevelSet.build_level_set_from_cfi"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        # only kernelpack-internal dotted prefixes should appear
        for ref in meta.cross_refs:
            first_part = ref.split(".")[0]
            assert first_part not in {"np", "scipy", "os", "sys", "math", "pathlib", "json"}

    def test_cross_refs_unique(self, symbol_table, geo_tree):
        method = _find_method(geo_tree, "RBFLevelSet", "build_level_set_from_cfi")
        src = str(SRC_ROOT / "geometry" / "core.py")
        qualname = "geometry.core.RBFLevelSet.build_level_set_from_cfi"
        meta = extract_metadata(
            _make_chunk(src, "geometry.core", qualname, method),
            symbol_table,
        )
        assert len(meta.cross_refs) == len(set(meta.cross_refs))
