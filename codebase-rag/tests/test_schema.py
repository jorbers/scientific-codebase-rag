"""Tests for schema.py — run with: pytest tests/test_schema.py -v

Uses QdrantClient(":memory:") so no running Qdrant instance is needed.
All confirmed dims are hardcoded here. If schema.py uses different dims,
these tests fail — that is the point.
"""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance

from kernelpack_rag.schema import COLLECTIONS_CONFIG, ensure_collections

CODE_COLLECTION = "kernelpack_code"
PAPERS_COLLECTION = "kernelpack_papers"


@pytest.fixture
def client():
    return QdrantClient(":memory:")


@pytest.fixture
def seeded_client(client):
    """Client with collections already created."""
    ensure_collections(client)
    return client


# ---------------------------------------------------------------------------
# Collection creation
# ---------------------------------------------------------------------------

class TestCollectionCreation:

    def test_creates_both_collections(self, client):
        ensure_collections(client)
        names = [c.name for c in client.get_collections().collections]
        assert CODE_COLLECTION in names
        assert PAPERS_COLLECTION in names

    def test_idempotent_second_run_does_not_raise(self, seeded_client):
        ensure_collections(seeded_client)  # must not raise


# ---------------------------------------------------------------------------
# kernelpack_code — dense spaces
# ---------------------------------------------------------------------------

EXPECTED_CODE_SPACES = {
    "ctx__jinacode":    896,
    "ctx__qwen3":       1024,
    "ctx__unixcoder":   768,
    "code__jinacode":   896,
    "code__qwen3":      1024,
    "code__unixcoder":  768,
    "codecom__jinacode": 896,
    "codecom__qwen3":   1024,
    "codecom__unixcoder": 768,
    "com__jinacode":    896,
    "com__qwen3":       1024,
    "com__unixcoder":   768,
    "math__qwen3":      1024,
    "summary__qwen3":   1024,
}


class TestCodeCollectionDenseSpaces:

    def test_all_spaces_present(self, seeded_client):
        info = seeded_client.get_collection(CODE_COLLECTION)
        declared = info.config.params.vectors
        for name in EXPECTED_CODE_SPACES:
            assert name in declared, f"Missing named space: {name}"

    def test_all_dims_correct(self, seeded_client):
        info = seeded_client.get_collection(CODE_COLLECTION)
        declared = info.config.params.vectors
        for name, expected_dim in EXPECTED_CODE_SPACES.items():
            actual_dim = declared[name].size
            assert actual_dim == expected_dim, (
                f"{name}: expected dim {expected_dim}, got {actual_dim}. "
                f"Wrong dim at write-once step = collection rebuild required."
            )

    def test_all_spaces_cosine_distance(self, seeded_client):
        info = seeded_client.get_collection(CODE_COLLECTION)
        for name, params in info.config.params.vectors.items():
            assert params.distance == Distance.COSINE, (
                f"{name} distance is not cosine"
            )

    def test_no_extra_undeclared_spaces(self, seeded_client):
        info = seeded_client.get_collection(CODE_COLLECTION)
        declared = set(info.config.params.vectors.keys())
        expected = set(EXPECTED_CODE_SPACES.keys())
        extra = declared - expected
        assert not extra, f"Undeclared spaces found in collection: {extra}"


# ---------------------------------------------------------------------------
# kernelpack_code — sparse space
# ---------------------------------------------------------------------------

class TestCodeCollectionSparseSpace:

    def test_bm25_code_declared(self, seeded_client):
        info = seeded_client.get_collection(CODE_COLLECTION)
        sparse = info.config.params.sparse_vectors
        assert sparse is not None
        assert "bm25_code" in sparse

    def test_no_extra_sparse_spaces(self, seeded_client):
        info = seeded_client.get_collection(CODE_COLLECTION)
        sparse = info.config.params.sparse_vectors or {}
        assert set(sparse.keys()) == {"bm25_code"}


# ---------------------------------------------------------------------------
# kernelpack_papers
# ---------------------------------------------------------------------------

class TestPapersCollection:

    def test_paper_qwen3_space_present(self, seeded_client):
        info = seeded_client.get_collection(PAPERS_COLLECTION)
        spaces = info.config.params.vectors
        assert "paper__qwen3" in spaces

    def test_paper_qwen3_dim(self, seeded_client):
        info = seeded_client.get_collection(PAPERS_COLLECTION)
        dim = info.config.params.vectors["paper__qwen3"].size
        assert dim == 1024

    def test_paper_qwen3_cosine(self, seeded_client):
        info = seeded_client.get_collection(PAPERS_COLLECTION)
        dist = info.config.params.vectors["paper__qwen3"].distance
        assert dist == Distance.COSINE

    def test_bm25_paper_declared(self, seeded_client):
        info = seeded_client.get_collection(PAPERS_COLLECTION)
        sparse = info.config.params.sparse_vectors
        assert sparse is not None
        assert "bm25_paper" in sparse


# ---------------------------------------------------------------------------
# Schema mismatch detection
# ---------------------------------------------------------------------------

class TestSchemaMismatchDetection:

    def test_wrong_dim_raises_value_error(self, seeded_client):
        """Simulate a stale collection with wrong dim — must hard-fail, not proceed."""
        import copy
        from unittest.mock import patch

        bad_config = copy.deepcopy(COLLECTIONS_CONFIG)
        bad_config[CODE_COLLECTION]["vectors"]["ctx__jinacode"]["size"] = 999

        with patch("kernelpack_rag.schema.COLLECTIONS_CONFIG", bad_config):
            with pytest.raises(ValueError):
                ensure_collections(seeded_client)

    def test_missing_space_raises_value_error(self, seeded_client):
        """Simulate config that dropped a space — must hard-fail."""
        import copy
        from unittest.mock import patch

        bad_config = copy.deepcopy(COLLECTIONS_CONFIG)
        del bad_config[CODE_COLLECTION]["vectors"]["math__qwen3"]

        with patch("kernelpack_rag.schema.COLLECTIONS_CONFIG", bad_config):
            with pytest.raises(ValueError):
                ensure_collections(seeded_client)

    def test_error_message_identifies_mismatch(self, seeded_client):
        """Error message must name what mismatched — not a generic failure."""
        import copy
        from unittest.mock import patch

        bad_config = copy.deepcopy(COLLECTIONS_CONFIG)
        bad_config[CODE_COLLECTION]["vectors"]["ctx__jinacode"]["size"] = 999

        with patch("kernelpack_rag.schema.COLLECTIONS_CONFIG", bad_config):
            with pytest.raises(ValueError, match="ctx__jinacode"):
                ensure_collections(seeded_client)
