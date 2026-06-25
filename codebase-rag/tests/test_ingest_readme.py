"""Integration tests for README ingestion — hits the real Qdrant instance."""

from __future__ import annotations

import pytest

from kernelpack_rag.config import CODE_COLLECTION, make_client
from kernelpack_rag.ingest_readme import README_PATH, ingest_readme, split_readme

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def readme_ingestion():
    """Run the ingest once and return (point_ids, n_sections)."""
    text = README_PATH.read_text()
    n_sections = len(split_readme(text))
    point_ids = ingest_readme(README_PATH)
    return point_ids, n_sections


@pytest.fixture(scope="module")
def retrieved_points(readme_ingestion):
    point_ids, _ = readme_ingestion
    qdrant = make_client()
    return qdrant.retrieve(
        collection_name=CODE_COLLECTION,
        ids=point_ids,
        with_payload=True,
        with_vectors=["ctx__jinacode", "bm25_code"],
    )


def test_all_points_found(readme_ingestion, retrieved_points):
    point_ids, n_sections = readme_ingestion
    # After paragraph sub-splitting, total chunks >= raw section count
    assert len(retrieved_points) >= n_sections
    assert len(retrieved_points) == len(point_ids)


def test_source_file_payload(retrieved_points):
    for point in retrieved_points:
        assert (point.payload or {}).get("source_file") == "README.md"


def test_ctx_jinacode_vector(retrieved_points):
    for point in retrieved_points:
        vectors = point.vector or {}
        dense = vectors.get("ctx__jinacode")
        assert dense is not None, f"Point {point.id} missing ctx__jinacode"
        assert len(dense) > 0, f"Point {point.id} has empty ctx__jinacode"


def test_bm25_code_vector(retrieved_points):
    for point in retrieved_points:
        vectors = point.vector or {}
        sparse = vectors.get("bm25_code")
        assert sparse is not None, f"Point {point.id} missing bm25_code"
        assert len(sparse.indices) > 0, f"Point {point.id} has empty bm25_code indices"
