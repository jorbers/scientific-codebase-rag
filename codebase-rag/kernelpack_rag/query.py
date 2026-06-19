"""CLI query interface for KernelPack RAG."""

from __future__ import annotations

import argparse

from qdrant_client import QdrantClient

from kernelpack_rag.config import CODE_COLLECTION, COLLECTIONS_CONFIG, make_client, PAPERS_COLLECTION
from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
from kernelpack_rag.retrieve import (
    Candidate,
    fine_to_coarse,
    hybrid,
    trimodal,
    two_leg,
)


DEFAULT_PLAN = "hybrid"
DEFAULT_K = 10
DEFAULT_SPACE = "ctx__jinacode"

PLAN_CHOICES = ("hybrid", "trimodal", "two_leg", "fine_to_coarse")

_CODE_SPACES = list(COLLECTIONS_CONFIG["kernelpack_code"]["vectors"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the KernelPack RAG index.",
    )
    parser.add_argument("query_text", help="The query string.")
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of results to return (default: %(default)s).",
    )
    parser.add_argument(
        "--plan",
        choices=PLAN_CHOICES,
        default=DEFAULT_PLAN,
        help="Retrieval plan to use (default: %(default)s).",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        default=[DEFAULT_SPACE],
        metavar="SPACE",
        choices=_CODE_SPACES,
        help=(
            f"Dense vector space(s) for the retrieval leg. "
            f"First value is used for single-space plans. "
            f"Default: {DEFAULT_SPACE}."
        ),
    )
    return parser.parse_args(argv)


def _run_plan(
    args: argparse.Namespace,
    client: QdrantClient,
) -> list[Candidate]:
    space = args.spaces[0]

    if args.plan == "hybrid":
        embedder = JinaCodeEmbedder()
        return hybrid(
            args.query_text,
            client=client,
            collection=CODE_COLLECTION,
            embedder=embedder,
            k=args.k,
            space=space,
        )

    if args.plan == "fine_to_coarse":
        embedder = JinaCodeEmbedder()
        return fine_to_coarse(
            args.query_text,
            client=client,
            collection=CODE_COLLECTION,
            embedder=embedder,
            k=args.k,
        )

    if args.plan == "trimodal":
        from kernelpack_rag.embed.qwen import QwenEmbedder

        jinacode_embedder = JinaCodeEmbedder()
        qwen_embedder = QwenEmbedder()
        return trimodal(
            args.query_text,
            client=client,
            collection=CODE_COLLECTION,
            jinacode_embedder=jinacode_embedder,
            qwen_embedder=qwen_embedder,
            weights={"ctx": 1.0, "math": 1.0, "sparse": 1.0},
            k=args.k,
        )

    if args.plan == "two_leg":
        embedder = JinaCodeEmbedder()
        return two_leg(
            args.query_text,
            client=client,
            code_collection=CODE_COLLECTION,
            papers_collection=PAPERS_COLLECTION,
            embedder=embedder,
            k=args.k,
        )

    return []


def _print_results(candidates: list[Candidate]) -> None:
    if not candidates:
        print("No results.")
        return

    for candidate in candidates:
        payload = candidate.payload
        function_name = payload.get("function_name") or "<unknown>"
        module = payload.get("module") or "<unknown>"
        text = str(payload.get("text") or "")[:200]
        print(
            f"[{candidate.fused_rank + 1}] {function_name}"
            f"  |  {module}"
            f"  |  score={candidate.fused_score:.4f}"
        )
        print(f"    {text}")
        print()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = make_client()
    candidates = _run_plan(args, client)
    _print_results(candidates)


if __name__ == "__main__":
    main()
