"""Verification gate for KernelPack RAG collection health and golden parity."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from kernelpack_rag.config import CODE_COLLECTION, PAPERS_COLLECTION, make_client
from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
from kernelpack_rag.qdrant_utils import _field_equals_filter, _scroll_points
from kernelpack_rag.rerank import NoopReranker
from kernelpack_rag.retrieve import hybrid
from kernelpack_rag.schema import ensure_collections


BASELINE_RECALL_AT_5 = 8 / 10
_RETRIEVE_BATCH_SIZE = 256


@dataclass
class InvariantResult:
    schema_ok: bool
    missing_primary_vectors: int
    unresolvable_parent_ids: int
    unresolvable_math_source_ids: int
    passed: bool


@dataclass
class ParityResult:
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    hits_at_3: int
    hits_at_5: int
    hits_at_10: int
    total: int
    gate_passed: bool
    per_query: list[dict]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify KernelPack RAG collection invariants and golden parity."
    )
    parser.add_argument(
        "--qa-pairs",
        required=True,
        type=Path,
        help="Path to the golden query set JSON file.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the package directory named kernelpack.",
    )
    return parser.parse_args(argv)


def matches(source_symbol: str, payload: dict) -> bool:
    function_name = payload.get("function_name", "")
    parent_class = payload.get("parent_class") or ""

    if "." in source_symbol:
        class_part, method_part = source_symbol.split(".", 1)
        return function_name == method_part and parent_class == class_part
    return function_name == source_symbol


def verify_invariants(client: QdrantClient) -> InvariantResult:
    try:
        ensure_collections(client)
        schema_ok = True
    except Exception as exc:
        print(f"Schema check error: {exc}")
        schema_ok = False

    missing_primary_vectors = sum(
        1
        for point in _scroll_points(
            client,
            CODE_COLLECTION,
            scroll_filter=_field_equals_filter("granularity", "coarse"),
            with_payload=False,
            with_vectors=["ctx__jinacode"],
        )
        if not _point_has_vector(point, "ctx__jinacode")
    )

    parent_ids = [
        (point.payload or {}).get("parent_id")
        for point in _scroll_points(
            client,
            CODE_COLLECTION,
            scroll_filter=_field_equals_filter("granularity", "fine"),
            with_payload=True,
            with_vectors=False,
        )
    ]
    unresolvable_parent_ids = _count_unresolvable_ids(
        client,
        CODE_COLLECTION,
        parent_ids,
    )

    unresolvable_math_source_ids = 0
    if _collection_has_points(client, PAPERS_COLLECTION):
        math_source_ids = [
            payload["math_source_id"]
            for point in _scroll_points(
                client,
                CODE_COLLECTION,
                scroll_filter=_field_equals_filter("granularity", "coarse"),
                with_payload=True,
                with_vectors=False,
            )
            if (payload := (point.payload or {})).get("math_source_id") is not None
        ]
        unresolvable_math_source_ids = _count_unresolvable_ids(
            client,
            PAPERS_COLLECTION,
            math_source_ids,
        )

    passed = (
        schema_ok
        and missing_primary_vectors == 0
        and unresolvable_parent_ids == 0
        and unresolvable_math_source_ids == 0
    )
    return InvariantResult(
        schema_ok=schema_ok,
        missing_primary_vectors=missing_primary_vectors,
        unresolvable_parent_ids=unresolvable_parent_ids,
        unresolvable_math_source_ids=unresolvable_math_source_ids,
        passed=passed,
    )


def verify_parity(
    client: QdrantClient,
    embedder: JinaCodeEmbedder,
    qa_pairs_path: Path,
) -> ParityResult:
    qa_pairs = json.loads(qa_pairs_path.read_text())

    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    per_query: list[dict] = []

    for qa_pair in qa_pairs:
        query = qa_pair["query"]
        source_symbol = qa_pair["source_symbol"]
        candidates = hybrid(
            query,
            client=client,
            collection=CODE_COLLECTION,
            embedder=embedder,
            k=10,
            reranker=NoopReranker(),
        )

        hit_at_3 = any(
            matches(source_symbol, candidate.payload) for candidate in candidates[:3]
        )
        hit_at_5 = any(
            matches(source_symbol, candidate.payload) for candidate in candidates[:5]
        )
        hit_at_10 = any(
            matches(source_symbol, candidate.payload) for candidate in candidates[:10]
        )

        hits_at_3 += int(hit_at_3)
        hits_at_5 += int(hit_at_5)
        hits_at_10 += int(hit_at_10)
        per_query.append(
            {
                "query": query,
                "tier": qa_pair["tier"],
                "source_symbol": source_symbol,
                "hit_at_5": hit_at_5,
                "top5_functions": [
                    _function_name(candidate.payload) for candidate in candidates[:5]
                ],
            }
        )

    total = len(qa_pairs)
    recall_at_3 = _rate(hits_at_3, total)
    recall_at_5 = _rate(hits_at_5, total)
    recall_at_10 = _rate(hits_at_10, total)

    return ParityResult(
        recall_at_3=recall_at_3,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        hits_at_3=hits_at_3,
        hits_at_5=hits_at_5,
        hits_at_10=hits_at_10,
        total=total,
        gate_passed=recall_at_5 >= BASELINE_RECALL_AT_5,
        per_query=per_query,
    )


def print_report(
    invariant_result: InvariantResult,
    parity_result: ParityResult,
) -> None:
    print("=== Collection Invariants ===")
    _print_check("Schema match:", invariant_result.schema_ok)
    _print_count_check(
        "Missing primary vectors:",
        invariant_result.missing_primary_vectors,
    )
    _print_count_check(
        "Unresolvable parent_ids:",
        invariant_result.unresolvable_parent_ids,
    )
    _print_count_check(
        "Unresolvable math_source_ids:",
        invariant_result.unresolvable_math_source_ids,
    )
    _print_check("Invariant gate:", invariant_result.passed)
    print()

    print(f"=== Golden-set Parity (n={parity_result.total}) ===")
    print(
        f"recall@3:  {_fraction(parity_result.hits_at_3, parity_result.total)}  "
        f"({parity_result.recall_at_3:.2f})"
    )
    print(
        f"recall@5:  {_fraction(parity_result.hits_at_5, parity_result.total)}  "
        f"({parity_result.recall_at_5:.2f})   <- gate threshold: "
        f"{BASELINE_RECALL_AT_5:.2f}"
    )
    print(
        f"recall@10: {_fraction(parity_result.hits_at_10, parity_result.total)}  "
        f"({parity_result.recall_at_10:.2f})"
    )
    print()
    print(f"Baseline recall@5: 8/10 ({BASELINE_RECALL_AT_5:.2f})")
    if parity_result.gate_passed:
        print("Result: PASS - no regression")
    else:
        print("Result: FAIL - recall@5 regression")
    print()

    print("Per-query breakdown:")
    for item in parity_result.per_query:
        top5 = ", ".join(item["top5_functions"])
        print(f'  [{item["tier"]:<11}] "{item["query"]}"')
        print(
            f'               source_symbol={item["source_symbol"]}   '
            f'hit@5={item["hit_at_5"]}   top5: [{top5}]'
        )
    print()

    print("=== Overall ===")
    if invariant_result.passed and parity_result.gate_passed:
        print("PASS - pipeline is ready")
    else:
        failed = []
        if not invariant_result.passed:
            failed.append("collection invariants")
        if not parity_result.gate_passed:
            failed.append("golden-set parity")
        print(f"FAIL - failed gates: {', '.join(failed)}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = make_client()
    embedder = JinaCodeEmbedder()

    invariant_result = verify_invariants(client)
    parity_result = verify_parity(client, embedder, Path(args.qa_pairs))

    print_report(invariant_result, parity_result)

    if not invariant_result.passed or not parity_result.gate_passed:
        sys.exit(1)


def _point_has_vector(point, space: str) -> bool:
    vectors = getattr(point, "vector", None)
    if not isinstance(vectors, dict):
        return False
    vector = vectors.get(space)
    return bool(vector)


def _collection_has_points(client: QdrantClient, collection_name: str) -> bool:
    return any(
        _scroll_points(
            client,
            collection_name,
            with_payload=False,
            with_vectors=False,
            limit=1,
        )
    )


def _count_unresolvable_ids(
    client: QdrantClient,
    collection_name: str,
    raw_ids: list[object],
) -> int:
    ids = [str(raw_id) for raw_id in raw_ids if raw_id is not None and raw_id != ""]
    missing_ids = len(raw_ids) - len(ids)
    if not ids:
        return missing_ids

    resolved_ids = _retrieve_existing_ids(client, collection_name, ids)
    return missing_ids + sum(1 for point_id in ids if point_id not in resolved_ids)


def _retrieve_existing_ids(
    client: QdrantClient,
    collection_name: str,
    ids: list[str],
) -> set[str]:
    resolved_ids: set[str] = set()
    unique_ids = sorted(set(ids))
    for start in range(0, len(unique_ids), _RETRIEVE_BATCH_SIZE):
        batch = unique_ids[start : start + _RETRIEVE_BATCH_SIZE]
        points = client.retrieve(
            collection_name=collection_name,
            ids=batch,
            with_payload=False,
            with_vectors=False,
        )
        resolved_ids.update(str(point.id) for point in points)
    return resolved_ids


def _function_name(payload: dict) -> str:
    name = payload.get("function_name")
    if name:
        return str(name)
    source = payload.get("source_file", "")
    chunk_type = payload.get("chunk_type", "")
    return f"{source}:{chunk_type}" if source or chunk_type else "<unknown>"


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def _print_check(label: str, passed: bool) -> None:
    print(f"{label:<32}{_status(passed)}")


def _print_count_check(label: str, count: int) -> None:
    print(f"{label:<32}{count:<8}{_status(count == 0)}")


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


if __name__ == "__main__":
    main()
