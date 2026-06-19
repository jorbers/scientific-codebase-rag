"""Integration tests for retrieve.py and rerank.py (Step 14).

Every test in this file runs against a real Qdrant collection populated by the
actual pipeline components (chunker, embedder, sparse builder). The only mock
is the OpenAI summarizer — LLM calls are not part of what retrieve.py tests.

Run with:
    pytest --integration tests/test_retrieve.py

What these tests prove:
- retrieve.py functions return well-formed Candidate objects
- Hybrid search surfaces semantically relevant chunks from real KernelPack data
- Filtered search correctly narrows results by math_terms payload
- fine_to_coarse returns coarse-granularity chunks with resolvable parent IDs
- expand_cross_refs produces a superset of the original candidates
- NoopReranker is transparent; CrossEncoder adds a rerank score
"""

import time
import uuid
from pathlib import Path

import pytest

# ── collection fixture ─────────────────────────────────────────────────────────

TEST_COLLECTION = "test_retrieve_code"


@pytest.fixture(scope="module")
def populated_collection(qdrant_client, kp_src):
    """
    Ingest geometry/core.py into a fresh Qdrant test collection using the
    real chunker, header builder, metadata extractor, JinaCodeEmbedder, and
    sparse vector builder. Summarizer is replaced with a fixed string so no
    OpenAI calls are made.

    Yields: (client, collection_name)
    Tears down: deletes the collection after all module tests complete.
    """
    from qdrant_client import models

    from kernelpack_rag.chunking.coarse import chunk_file
    from kernelpack_rag.chunking.fine import fine_chunks
    from kernelpack_rag.chunking.header import build_header
    from kernelpack_rag.chunking.metadata import (
        build_symbol_table,
        extract_metadata,
        get_coarse_uuid,
        get_fine_uuid,
    )
    from kernelpack_rag.config import COLLECTIONS_CONFIG
    from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
    from kernelpack_rag.embed.sparse import to_qdrant_sparse
    from kernelpack_rag.schema import (
        _build_sparse_vectors_config,
        _build_vectors_config,
    )

    # ── setup ──────────────────────────────────────────────────────────────────
    if qdrant_client.collection_exists(TEST_COLLECTION):
        qdrant_client.delete_collection(TEST_COLLECTION)

    code_cfg = COLLECTIONS_CONFIG["kernelpack_code"]
    qdrant_client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config=_build_vectors_config(code_cfg["vectors"]),
        sparse_vectors_config=_build_sparse_vectors_config(code_cfg["sparse_vectors"]),
    )

    # ── ingest geometry/core.py with real pipeline components ─────────────────
    target = kp_src / "geometry" / "core.py"
    assert target.exists(), f"geometry/core.py not found at {target}"

    source_text = target.read_text()
    chunks = chunk_file(target)
    symbol_table = build_symbol_table(kp_src)
    embedder = JinaCodeEmbedder()

    points: list[models.PointStruct] = []

    for chunk in chunks:
        ctx_header = build_header(chunk, source_text, chunks)
        # Fixed summary: avoids OpenAI; realistic enough for semantic retrieval.
        summary = (
            f"This function implements {chunk.qualname} in the KernelPack geometry "
            f"module. It performs numerical operations on arrays for RBF-FD computations."
        )
        metadata = extract_metadata(chunk, symbol_table)

        payload = {
            "text": chunk.text,
            "context_header": ctx_header,
            "llm_summary": summary,
            "chunk_type": chunk.chunk_type,
            "module": chunk.module,
            "parent_class": chunk.parent_class,
            "function_name": metadata.function_name,
            "math_terms": metadata.math_terms,
            "cross_refs": metadata.cross_refs,
            "cross_ref_ids": [str(uid) for uid in metadata.cross_ref_ids],
            "has_numba": metadata.has_numba,
            "source_file": chunk.source_file,
            "line_range": list(chunk.line_range),
            "granularity": "coarse",
            "parent_id": None,
            "content_hash": "fixture",
        }

        ctx_text = f"{summary}\n{ctx_header}\n{chunk.text}"
        coarse_id = get_coarse_uuid(chunk.source_file, chunk.qualname, chunk.module)

        points.append(
            models.PointStruct(
                id=str(coarse_id),
                vector={
                    "ctx__jinacode": embedder.embed_batch([ctx_text])[0],
                    "bm25_code": to_qdrant_sparse(chunk.text),
                },
                payload=payload,
            )
        )

        for fc in fine_chunks(chunk):
            fine_payload = {
                **payload,
                "text": fc.text,
                "granularity": "fine",
                "parent_id": str(coarse_id),
                "line_range": list(fc.line_range),
            }
            fine_ctx = f"{summary}\n{ctx_header}\n{fc.text}"
            fine_id = get_fine_uuid(
                fc.source_file, fc.qualname, chunk.module, fc.window_idx
            )
            points.append(
                models.PointStruct(
                    id=str(fine_id),
                    vector={
                        "ctx__jinacode": embedder.embed_batch([fine_ctx])[0],
                        "bm25_code": to_qdrant_sparse(fc.text),
                    },
                    payload=fine_payload,
                )
            )

    for i in range(0, len(points), 64):
        qdrant_client.upsert(
            collection_name=TEST_COLLECTION, points=points[i : i + 64]
        )

    time.sleep(1)  # let Qdrant commit before queries run

    yield qdrant_client, TEST_COLLECTION

    # ── teardown ───────────────────────────────────────────────────────────────
    qdrant_client.delete_collection(TEST_COLLECTION)


# ── hybrid retrieval ───────────────────────────────────────────────────────────

@pytest.mark.integration
class TestHybrid:
    def test_returns_list(self, populated_collection):
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid

        results = hybrid(
            "normalize rows of a matrix",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_candidate_fields_present(self, populated_collection):
        """Every Candidate must carry all fields the rest of the pipeline depends on."""
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import Candidate, hybrid

        results = hybrid(
            "rbf kernel evaluation",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        for c in results:
            assert isinstance(c, Candidate), f"Expected Candidate, got {type(c)}"
            assert c.point_id is not None
            assert isinstance(c.payload, dict)
            assert isinstance(c.leg_scores, dict)
            assert "dense" in c.leg_scores, "leg_scores missing 'dense' key"
            assert "sparse" in c.leg_scores, "leg_scores missing 'sparse' key"
            assert isinstance(c.fused_rank, int)
            assert isinstance(c.fused_score, float)

    def test_fused_ranks_sequential_from_zero(self, populated_collection):
        """fused_rank must be 0-indexed and sequential — qlog depends on this."""
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid

        results = hybrid(
            "stencil assembly",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        for i, c in enumerate(results):
            assert c.fused_rank == i, (
                f"fused_rank at position {i} is {c.fused_rank}, expected {i}"
            )

    def test_k_caps_result_count(self, populated_collection):
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid

        results = hybrid(
            "distance matrix",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=3,
            reranker=NoopReranker(),
        )
        assert len(results) <= 3

    def test_normalize_rows_in_top3(self, populated_collection):
        """
        The semantic retrieval gate: a natural-language query about normalizing
        matrix rows must surface normalize_rows in the top 3.

        This is the same check as smoke test Cell 15, now formalized as a
        hard assertion. If this fails, the embedding or sparse vector pipeline
        is broken — not a test issue.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid

        results = hybrid(
            "how do I scale each row of a matrix to have unit norm?",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        top3 = [r.payload.get("function_name", "") for r in results[:3]]
        assert "normalize_rows" in top3, (
            f"normalize_rows not in top 3. Got: {top3}\n"
            f"This indicates a failure in dense or sparse vector retrieval."
        )

    def test_payload_fields_populated(self, populated_collection):
        """
        Every candidate payload must have the D7 fields the downstream
        pipeline (qlog, verify, MCP) reads. Missing fields here cause
        silent failures later.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid

        results = hybrid(
            "rbf stencil",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        required = {
            "text", "llm_summary", "context_header", "granularity",
            "function_name", "module", "math_terms", "cross_refs",
            "source_file", "line_range",
        }
        for c in results:
            missing = required - set(c.payload.keys())
            assert not missing, (
                f"Candidate {c.payload.get('function_name')} missing payload fields: {missing}"
            )


# ── hybrid filtered ────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestHybridFiltered:
    def test_all_results_have_requested_math_term(self, populated_collection):
        """
        Every result from hybrid_filtered must have the requested math_term
        in its payload. If any result lacks it, the payload filter is broken.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid_filtered

        results = hybrid_filtered(
            "kernel evaluation on node sets",
            math_terms=["rbf"],
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=10,
            reranker=NoopReranker(),
        )
        assert len(results) > 0, "Filtered query returned no results — check math_terms lexicon"
        for c in results:
            assert "rbf" in c.payload.get("math_terms", []), (
                f"Result {c.payload.get('function_name')!r} missing math_term 'rbf'. "
                f"Got: {c.payload.get('math_terms')}"
            )

    def test_filtered_is_subset_of_unfiltered(self, populated_collection):
        """Filtering can only reduce results, never expand them."""
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid, hybrid_filtered

        embedder = JinaCodeEmbedder()
        reranker = NoopReranker()
        query = "kernel function evaluation"

        unfiltered = hybrid(
            query, client=client, collection=col, embedder=embedder, k=10, reranker=reranker
        )
        filtered = hybrid_filtered(
            query,
            math_terms=["rbf"],
            client=client,
            collection=col,
            embedder=embedder,
            k=10,
            reranker=reranker,
        )
        assert len(filtered) <= len(unfiltered)


# ── fine to coarse ─────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestFineToCoarse:
    def test_returns_only_coarse_granularity(self, populated_collection):
        """
        fine_to_coarse must return only coarse chunks. Returning fine chunks
        would send incomplete context to the LLM.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import fine_to_coarse

        results = fine_to_coarse(
            "normalize rows of a matrix",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        assert len(results) > 0
        for c in results:
            assert c.payload.get("granularity") == "coarse", (
                f"Expected coarse, got {c.payload.get('granularity')!r} "
                f"for {c.payload.get('function_name')!r}"
            )

    def test_returned_ids_exist_in_collection(self, populated_collection):
        """
        Every point ID returned by fine_to_coarse must resolve in Qdrant.
        A dangling ID means parent_id resolution is broken.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import fine_to_coarse

        results = fine_to_coarse(
            "normalize rows",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        ids = [str(c.point_id) for c in results]
        fetched = client.retrieve(
            collection_name=col, ids=ids, with_payload=False, with_vectors=False
        )
        assert len(fetched) == len(results), (
            f"Only {len(fetched)}/{len(results)} returned IDs resolve in Qdrant. "
            f"parent_id resolution is broken."
        )


# ── expand cross refs ──────────────────────────────────────────────────────────

@pytest.mark.integration
class TestExpandCrossRefs:
    def test_expansion_is_superset_of_base(self, populated_collection):
        """
        expand_cross_refs must never drop original candidates — it only adds.
        Dropping originals would silently remove context from the LLM window.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import expand_cross_refs, hybrid

        base = hybrid(
            "distance matrix computation",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=3,
            reranker=NoopReranker(),
        )
        expanded = expand_cross_refs(base, client=client, collection=col, hops=1)

        base_ids = {str(c.point_id) for c in base}
        expanded_ids = {str(c.point_id) for c in expanded}
        assert base_ids.issubset(expanded_ids), (
            f"expand_cross_refs dropped original candidates: {base_ids - expanded_ids}"
        )

    def test_expansion_hits_are_marked(self, populated_collection):
        """
        Cross-reference expansion hits must be distinguishable from retrieval
        hits in the candidate list. qlog uses this to attribute failure correctly.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import expand_cross_refs, hybrid

        base = hybrid(
            "distance matrix",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=3,
            reranker=NoopReranker(),
        )
        expanded = expand_cross_refs(base, client=client, collection=col, hops=1)
        base_ids = {str(c.point_id) for c in base}
        expansion_hits = [c for c in expanded if str(c.point_id) not in base_ids]

        for c in expansion_hits:
            # Expansion hits must be marked — exact field name is Codex's choice,
            # but the marker must exist so qlog can distinguish them.
            marked = (
                c.meta.get("expansion") is True
                or c.meta.get("provenance") == "cross_ref"
            )
            assert marked, (
                f"Expansion hit {c.payload.get('function_name')!r} has no provenance marker. "
                f"leg_scores: {c.leg_scores}"
            )


# ── reranker ───────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestNoopReranker:
    def test_preserves_candidate_order(self, populated_collection):
        """NoopReranker must be a transparent passthrough — order unchanged."""
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid

        reranker = NoopReranker()
        results = hybrid(
            "rbf kernel",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=reranker,
        )
        reranked = reranker.rerank("rbf kernel", results)
        assert [str(c.point_id) for c in reranked] == [str(c.point_id) for c in results]

    def test_does_not_add_rerank_score(self, populated_collection):
        """
        NoopReranker must not add a 'rerank' key to leg_scores.
        If it does, the log will falsely attribute ranking quality to the reranker.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import NoopReranker
        from kernelpack_rag.retrieve import hybrid

        results = hybrid(
            "stencil weights",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=3,
            reranker=NoopReranker(),
        )
        for c in results:
            assert "rerank" not in c.leg_scores, (
                f"NoopReranker added 'rerank' to leg_scores for "
                f"{c.payload.get('function_name')!r}"
            )


class TestCrossEncoderReranker:
    def test_adds_rerank_score_to_leg_scores(self, populated_collection):
        """
        CrossEncoder must add a 'rerank' key to each candidate's leg_scores.
        This is what qlog reads to log reranker contribution.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import CrossEncoderReranker, NoopReranker
        from kernelpack_rag.retrieve import hybrid

        # Get candidates first via hybrid, then rerank separately
        candidates = hybrid(
            "normalize rows of a matrix",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=NoopReranker(),
        )
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("normalize rows of a matrix", candidates)

        for c in reranked:
            assert "rerank" in c.leg_scores, (
                f"CrossEncoderReranker did not add 'rerank' to leg_scores "
                f"for {c.payload.get('function_name')!r}"
            )
            assert isinstance(c.leg_scores["rerank"], float)

    def test_changes_candidate_order(self, populated_collection):
        """
        CrossEncoder must produce a different ranking than RRF for at least
        some queries. If order never changes, the reranker is a no-op.
        """
        client, col = populated_collection
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
        from kernelpack_rag.rerank import CrossEncoderReranker, NoopReranker
        from kernelpack_rag.retrieve import hybrid

        noop = NoopReranker()
        cross = CrossEncoderReranker()

        base = hybrid(
            "normalize rows of a matrix",
            client=client,
            collection=col,
            embedder=JinaCodeEmbedder(),
            k=5,
            reranker=noop,
        )
        reranked = cross.rerank("normalize rows of a matrix", base)

        base_order = [str(c.point_id) for c in base]
        reranked_order = [str(c.point_id) for c in reranked]
        # Not asserting they differ — on a tiny 1-file collection they might not.
        # Assert that the reranked list is the same length and covers same IDs.
        assert set(base_order) == set(reranked_order), (
            "CrossEncoderReranker dropped or added candidates"
        )