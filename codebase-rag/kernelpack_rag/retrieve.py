"""Retrieval plans for KernelPack RAG."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

from qdrant_client import QdrantClient, models

from kernelpack_rag import qlog
from kernelpack_rag.embed.sparse import to_qdrant_sparse
from kernelpack_rag.qdrant_utils import _field_any_filter, _field_equals_filter


class CodeChunk(TypedDict):
    point_id: str
    module: str
    function_name: str
    chunk_type: str
    text: str
    llm_summary: str
    scores: dict[str, float]


@dataclass
class Candidate:
    point_id: str
    payload: dict
    leg_scores: dict[str, float]
    fused_rank: int
    fused_score: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        # payload is assumed to be JSON-serializable (strings, ints, lists, bools)
        return {
            "point_id": self.point_id,
            "payload": self.payload,
            "leg_scores": self.leg_scores,
            "fused_rank": self.fused_rank,
            "fused_score": self.fused_score,
            "meta": self.meta,
        }

    def to_code_chunk(self) -> CodeChunk:
        p = self.payload
        return CodeChunk(
            point_id=self.point_id,
            module=str(p.get("module") or ""),
            function_name=str(p.get("function_name") or ""),
            chunk_type=str(p.get("chunk_type") or ""),
            text=str(p.get("text") or ""),
            llm_summary=str(p.get("llm_summary") or ""),
            scores=dict(self.leg_scores),
        )


def hybrid(
    query: str,
    *,
    client: QdrantClient,
    collection: str,
    embedder,
    k: int = 10,
    space: str = "ctx__jinacode",
    reranker=None,
    injected_ids: list[str] | None = None,
    query_filter: models.Filter | None = None,
    query_id: str | None = None,
    log_path: Path | None = None,
) -> list[Candidate]:
    reranker_id = _reranker_id(reranker)
    if injected_ids:
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="hybrid", spaces=[space],
            fusion="rrf", filters={}, reranker_id=reranker_id,
            candidates=[], injected_ids=list(injected_ids),
        )
        return []

    candidates = _hybrid_query(
        query,
        client=client,
        collection=collection,
        embedder=embedder,
        dense_space=space,
        sparse_space="bm25_code",
        k=k,
        query_filter=query_filter,
        leg_names=("dense", "sparse"),
    )
    candidates = _apply_reranker(query, candidates, reranker)
    _log_retrieval(
        query_id, log_path,
        query_text=query, plan="hybrid", spaces=[space],
        fusion="rrf", filters={}, reranker_id=reranker_id,
        candidates=candidates, injected_ids=[],
    )
    return candidates


def hybrid_filtered(
    query: str,
    *,
    math_terms: list[str],
    client: QdrantClient,
    collection: str,
    embedder,
    k: int = 10,
    space: str = "ctx__jinacode",
    reranker=None,
    injected_ids: list[str] | None = None,
    query_id: str | None = None,
    log_path: Path | None = None,
) -> list[Candidate]:
    reranker_id = _reranker_id(reranker)
    if injected_ids:
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="hybrid_filtered", spaces=[space],
            fusion="rrf", filters={"math_terms": math_terms},
            reranker_id=reranker_id,
            candidates=[], injected_ids=list(injected_ids),
        )
        return []
    if not math_terms:
        return []

    candidates = _hybrid_query(
        query,
        client=client,
        collection=collection,
        embedder=embedder,
        dense_space=space,
        sparse_space="bm25_code",
        k=k,
        query_filter=_field_any_filter("math_terms", math_terms),
        leg_names=("dense", "sparse"),
    )
    candidates = _apply_reranker(query, candidates, reranker)
    _log_retrieval(
        query_id, log_path,
        query_text=query, plan="hybrid_filtered", spaces=[space],
        fusion="rrf", filters={"math_terms": math_terms},
        reranker_id=reranker_id,
        candidates=candidates, injected_ids=[],
    )
    return candidates


def trimodal(
    query: str,
    *,
    client: QdrantClient,
    collection: str,
    jinacode_embedder,
    qwen_embedder,
    weights: dict,
    k: int = 10,
    reranker=None,
    injected_ids: list[str] | None = None,
    query_id: str | None = None,
    log_path: Path | None = None,
) -> list[Candidate]:
    reranker_id = _reranker_id(reranker)
    trimodal_spaces = ["ctx__jinacode", "math__qwen3", "bm25_code"]
    if injected_ids:
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="trimodal", spaces=trimodal_spaces,
            fusion="weighted_rrf", filters={}, reranker_id=reranker_id,
            candidates=[], injected_ids=list(injected_ids),
        )
        return []

    prefetch_k = _prefetch_k(k)
    ctx_vector = jinacode_embedder.embed_query_batch([query])[0]
    math_vector = qwen_embedder.embed_query_batch([query])[0]
    sparse_vector = to_qdrant_sparse(query)

    legs = [
        (
            "dense",
            float(weights.get("ctx", 1.0)),
            _single_leg_query(
                client=client,
                collection=collection,
                query=ctx_vector,
                using="ctx__jinacode",
                k=prefetch_k,
            ),
        ),
        (
            "math",
            float(weights.get("math", 1.0)),
            _single_leg_query(
                client=client,
                collection=collection,
                query=math_vector,
                using="math__qwen3",
                k=prefetch_k,
            ),
        ),
        (
            "sparse",
            float(weights.get("sparse", 1.0)),
            _single_leg_query(
                client=client,
                collection=collection,
                query=models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values,
                ),
                using="bm25_code",
                k=prefetch_k,
            ),
        ),
    ]

    by_id: dict[str, Candidate] = {}
    for leg_name, weight, leg_candidates in legs:
        for rank, candidate in enumerate(leg_candidates):
            score = _rrf_score(rank, weight)
            existing = by_id.get(candidate.point_id)
            if existing is None:
                by_id[candidate.point_id] = Candidate(
                    point_id=candidate.point_id,
                    payload=candidate.payload,
                    leg_scores={leg_name: candidate.fused_score},
                    fused_rank=0,
                    fused_score=score,
                )
            else:
                existing.leg_scores[leg_name] = candidate.fused_score
                existing.fused_score += score

    candidates = sorted(
        by_id.values(),
        key=lambda candidate: candidate.fused_score,
        reverse=True,
    )
    candidates = candidates[:k]
    _renumber_ranks(candidates)
    candidates = _apply_reranker(query, candidates, reranker)
    _log_retrieval(
        query_id, log_path,
        query_text=query, plan="trimodal", spaces=trimodal_spaces,
        fusion="weighted_rrf", filters={}, reranker_id=reranker_id,
        candidates=candidates, injected_ids=[],
    )
    return candidates


def two_leg(
    query: str,
    *,
    client: QdrantClient,
    code_collection: str,
    papers_collection: str,
    embedder,
    k: int = 10,
    reranker=None,
    injected_ids: list[str] | None = None,
    query_id: str | None = None,
    log_path: Path | None = None,
) -> list[Candidate]:
    reranker_id = _reranker_id(reranker)
    two_leg_spaces = ["paper__qwen3", "bm25_paper", "ctx__jinacode", "bm25_code"]
    if injected_ids:
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="two_leg", spaces=two_leg_spaces,
            fusion="rrf", filters={}, reranker_id=reranker_id,
            candidates=[], injected_ids=list(injected_ids),
        )
        return []

    papers = _hybrid_query(
        query,
        client=client,
        collection=papers_collection,
        embedder=embedder,
        dense_space="paper__qwen3",
        sparse_space="bm25_paper",
        k=max(5, k),
        query_filter=None,
        leg_names=("dense", "sparse"),
    )
    bridge_papers = papers[:5]
    paper_bridge_ids = [candidate.point_id for candidate in bridge_papers]
    math_terms = sorted(
        {
            term
            for candidate in bridge_papers
            for term in candidate.payload.get("math_terms", [])
            if term
        }
    )
    if not math_terms:
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="two_leg", spaces=two_leg_spaces,
            fusion="rrf", filters={}, reranker_id=reranker_id,
            candidates=[], injected_ids=[],
        )
        return []

    code_candidates = hybrid_filtered(
        query,
        math_terms=math_terms,
        client=client,
        collection=code_collection,
        embedder=embedder,
        k=k,
        reranker=reranker,
    )
    for candidate in code_candidates:
        candidate.meta["paper_bridge"] = paper_bridge_ids
    _log_retrieval(
        query_id, log_path,
        query_text=query, plan="two_leg", spaces=two_leg_spaces,
        fusion="rrf", filters={}, reranker_id=reranker_id,
        candidates=code_candidates, injected_ids=[],
    )
    return code_candidates


def expand_cross_refs(
    candidates: list[Candidate],
    *,
    client: QdrantClient,
    collection: str,
    hops: int = 1,
    query_id: str | None = None,
    log_path: Path | None = None,
) -> list[Candidate]:
    if hops <= 0:
        return candidates

    expanded = list(candidates)
    seen = {candidate.point_id for candidate in expanded}
    frontier = list(candidates)

    for _ in range(hops):
        ids_to_fetch: list[str] = []
        for candidate in frontier:
            for point_id in candidate.payload.get("cross_ref_ids", []) or []:
                point_id = str(point_id)
                if point_id not in seen:
                    seen.add(point_id)
                    ids_to_fetch.append(point_id)

        if not ids_to_fetch:
            break

        try:
            fetched = client.retrieve(
                collection_name=collection,
                ids=ids_to_fetch,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            print(f"Qdrant retrieve error in expand_cross_refs: {exc}", file=sys.stderr)
            fetched = []

        next_frontier: list[Candidate] = []
        for point in fetched:
            candidate = Candidate(
                point_id=str(point.id),
                payload=point.payload or {},
                leg_scores={},
                fused_rank=len(expanded),
                fused_score=0.0,
                meta={"expansion": True, "provenance": "cross_ref"},
            )
            expanded.append(candidate)
            next_frontier.append(candidate)
        frontier = next_frontier

    _log_retrieval(
        query_id, log_path,
        query_text="", plan="expand_cross_refs", spaces=[],
        fusion="none", filters={}, reranker_id=None,
        candidates=expanded, injected_ids=[],
    )
    return expanded


def fine_to_coarse(
    query: str,
    *,
    client: QdrantClient,
    collection: str,
    embedder,
    k: int = 10,
    reranker=None,
    injected_ids: list[str] | None = None,
    query_id: str | None = None,
    log_path: Path | None = None,
) -> list[Candidate]:
    reranker_id = _reranker_id(reranker)
    ftc_spaces = ["ctx__jinacode", "bm25_code"]
    if injected_ids:
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="fine_to_coarse", spaces=ftc_spaces,
            fusion="rrf", filters={"granularity": "fine"}, reranker_id=reranker_id,
            candidates=[], injected_ids=list(injected_ids),
        )
        return []

    fine_candidates = _hybrid_query(
        query,
        client=client,
        collection=collection,
        embedder=embedder,
        dense_space="ctx__jinacode",
        sparse_space="bm25_code",
        k=_prefetch_k(k),
        query_filter=_field_equals_filter("granularity", "fine"),
        leg_names=("dense", "sparse"),
    )

    best_by_parent: dict[str, Candidate] = {}
    for candidate in fine_candidates:
        parent_id = candidate.payload.get("parent_id")
        if not parent_id:
            continue
        parent_id = str(parent_id)
        existing = best_by_parent.get(parent_id)
        if existing is None or candidate.fused_score > existing.fused_score:
            best_by_parent[parent_id] = candidate

    parent_ids = [
        parent_id
        for parent_id, _ in sorted(
            best_by_parent.items(),
            key=lambda item: item[1].fused_score,
            reverse=True,
        )
    ][:k]
    if not parent_ids:
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="fine_to_coarse", spaces=ftc_spaces,
            fusion="rrf", filters={"granularity": "fine"}, reranker_id=reranker_id,
            candidates=[], injected_ids=[],
        )
        return []

    try:
        fetched = client.retrieve(
            collection_name=collection,
            ids=parent_ids,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        print(f"Qdrant retrieve error in fine_to_coarse: {exc}", file=sys.stderr)
        _log_retrieval(
            query_id, log_path,
            query_text=query, plan="fine_to_coarse", spaces=ftc_spaces,
            fusion="rrf", filters={"granularity": "fine"}, reranker_id=reranker_id,
            candidates=[], injected_ids=[],
        )
        return []

    fetched_by_id = {str(point.id): point for point in fetched}

    candidates: list[Candidate] = []
    for parent_id in parent_ids:
        point = fetched_by_id.get(parent_id)
        if point is None:
            continue
        child = best_by_parent[parent_id]
        candidates.append(
            Candidate(
                point_id=parent_id,
                payload=point.payload or {},
                leg_scores=dict(child.leg_scores),
                fused_rank=len(candidates),
                fused_score=child.fused_score,
                meta={"child_id": child.point_id},
            )
        )

    candidates = _apply_reranker(query, candidates, reranker)
    _log_retrieval(
        query_id, log_path,
        query_text=query, plan="fine_to_coarse", spaces=ftc_spaces,
        fusion="rrf", filters={"granularity": "fine"}, reranker_id=reranker_id,
        candidates=candidates, injected_ids=[],
    )
    return candidates


def _hybrid_query(
    query: str,
    *,
    client: QdrantClient,
    collection: str,
    embedder,
    dense_space: str,
    sparse_space: str,
    k: int,
    query_filter: models.Filter | None,
    leg_names: tuple[str, str],
) -> list[Candidate]:
    dense_query_vector = embedder.embed_query_batch([query])[0]
    sparse_query = to_qdrant_sparse(query)
    prefetch_k = _prefetch_k(k)
    dense_leg_name, sparse_leg_name = leg_names

    # Query each leg separately to capture raw per-leg scores for logging.
    # Qdrant's server-side RRF returns only the fused score, so we must
    # probe each leg independently before fusing.
    shared_kwargs: dict[str, Any] = {
        "collection_name": collection,
        "limit": prefetch_k,
        "with_payload": False,
    }
    if query_filter is not None:
        shared_kwargs["query_filter"] = query_filter

    dense_leg_scores: dict[str, float] = {}
    sparse_leg_scores: dict[str, float] = {}

    try:
        for point in _result_points(
            client.query_points(query=dense_query_vector, using=dense_space, **shared_kwargs)
        ):
            dense_leg_scores[str(point.id)] = float(getattr(point, "score", 0.0) or 0.0)
    except Exception:
        pass

    try:
        for point in _result_points(
            client.query_points(
                query=models.SparseVector(
                    indices=sparse_query.indices,
                    values=sparse_query.values,
                ),
                using=sparse_space,
                **shared_kwargs,
            )
        ):
            sparse_leg_scores[str(point.id)] = float(getattr(point, "score", 0.0) or 0.0)
    except Exception:
        pass

    call_kwargs: dict[str, Any] = {
        "collection_name": collection,
        "prefetch": [
            models.Prefetch(
                query=dense_query_vector,
                using=dense_space,
                limit=prefetch_k,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_query.indices,
                    values=sparse_query.values,
                ),
                using=sparse_space,
                limit=prefetch_k,
            ),
        ],
        "query": models.FusionQuery(fusion=models.Fusion.RRF),
        "limit": k,
        "with_payload": True,
    }
    if query_filter is not None:
        call_kwargs["query_filter"] = query_filter

    try:
        results = client.query_points(**call_kwargs)
    except Exception as exc:
        print(f"Qdrant query_points error: {exc}", file=sys.stderr)
        return []

    candidates = []
    for rank, point in enumerate(_result_points(results)):
        pid = str(point.id)
        candidates.append(
            Candidate(
                point_id=pid,
                payload=point.payload or {},
                leg_scores={
                    dense_leg_name: dense_leg_scores.get(pid, 0.0),
                    sparse_leg_name: sparse_leg_scores.get(pid, 0.0),
                },
                fused_rank=rank,
                fused_score=float(getattr(point, "score", 0.0) or 0.0),
            )
        )
    return candidates


def _single_leg_query(
    *,
    client: QdrantClient,
    collection: str,
    query,
    using: str,
    k: int,
) -> list[Candidate]:
    try:
        results = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=query,
                    using=using,
                    limit=k,
                )
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
    except Exception as exc:
        print(f"Qdrant query_points error: {exc}", file=sys.stderr)
        return []
    return _points_to_candidates(_result_points(results), (using,))


def _points_to_candidates(points, leg_names: tuple[str, ...]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for rank, point in enumerate(points):
        score = float(getattr(point, "score", 0.0) or 0.0)
        candidates.append(
            Candidate(
                point_id=str(point.id),
                payload=point.payload or {},
                leg_scores={leg_name: score for leg_name in leg_names},
                fused_rank=rank,
                fused_score=score,
            )
        )
    return candidates


def _result_points(results):
    return getattr(results, "points", results)


def _apply_reranker(
    query: str,
    candidates: list[Candidate],
    reranker,
) -> list[Candidate]:
    if reranker is None:
        from kernelpack_rag.rerank import NoopReranker

        reranker = NoopReranker()

    reranked = reranker.rerank(query, candidates)
    _renumber_ranks(reranked)
    return reranked


def _renumber_ranks(candidates: list[Candidate]) -> None:
    for rank, candidate in enumerate(candidates):
        candidate.fused_rank = rank


def _prefetch_k(k: int) -> int:
    return max(k, 20)


def _rrf_score(rank: int, weight: float) -> float:
    return weight / (60.0 + rank + 1.0)


def _reranker_id(reranker) -> str:
    if reranker is None:
        return "NoopReranker"
    return getattr(reranker, "id", type(reranker).__name__)


def _log_retrieval(
    query_id: str | None,
    log_path: Path | None,
    *,
    query_text: str,
    plan: str,
    spaces: list[str],
    fusion: str,
    filters: dict,
    reranker_id: str | None,
    candidates: list[Candidate],
    injected_ids: list[str],
) -> None:
    if query_id is None or log_path is None:
        return
    qlog.write_event(log_path, {
        "query_id": query_id,
        "event": "retrieval",
        "payload": {
            "query_text": query_text,
            "plan": plan,
            "spaces": spaces,
            "fusion": fusion,
            "filters": filters,
            "reranker_id": reranker_id,
            "candidates": [
                {
                    "point_id": c.point_id,
                    "leg_scores": c.leg_scores,
                    "fused_rank": c.fused_rank,
                    "fused_score": c.fused_score,
                }
                for c in candidates
            ],
            "injected_ids": injected_ids,
        },
    })


__all__ = [
    "Candidate",
    "CodeChunk",
    "hybrid",
    "hybrid_filtered",
    "trimodal",
    "two_leg",
    "expand_cross_refs",
    "fine_to_coarse",
]
