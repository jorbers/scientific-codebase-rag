"""KernelPack RAG ingestion orchestrator."""

from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from openai import OpenAI
from qdrant_client import QdrantClient, models

from kernelpack_rag.chunking.coarse import CoarseChunk, chunk_file
from kernelpack_rag.chunking.fine import fine_chunks
from kernelpack_rag.chunking.header import build_header
from kernelpack_rag.chunking.metadata import (
    KP_NAMESPACE,
    build_symbol_table,
    extract_metadata,
    get_coarse_uuid,
    get_fine_uuid,
)
from kernelpack_rag.chunking.papers import PaperChunk, load_paper_chunks
from kernelpack_rag.config import CODE_COLLECTION, COLLECTIONS_CONFIG, make_client, PAPERS_COLLECTION
from kernelpack_rag.embed.base import EmbedderRegistry
from kernelpack_rag.embed.representations import RepresentationKey, build_representation
from kernelpack_rag.embed.sparse import to_qdrant_sparse
from kernelpack_rag.enrich.summarize import content_hash, summarize_chunk
from kernelpack_rag.schema import ensure_collections
from kernelpack_rag.qdrant_utils import _field_any_filter, _field_equals_filter, _scroll_points


BATCH_SIZE = 64
SUMMARY_CACHE_DIR = Path(__file__).parent.parent / "summaries_cache"
DEFAULT_CODE_SPACES = ("ctx__jinacode", "bm25_code", "math__qwen3")
_FINE_ONLY_SPACES = {"ctx__jinacode", "bm25_code"}
DEFAULT_PAPER_SPACES = ("paper__qwen3", "bm25_paper")


class UpsertBatcher:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        self.points: list[models.PointStruct] = []

    def add(self, point: models.PointStruct) -> None:
        self.points.append(point)
        if len(self.points) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self.points:
            return
        self.client.upsert(collection_name=self.collection_name, points=self.points)
        self.points = []


class VectorUpdateBatcher:
    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name
        self.points: list[models.PointVectors] = []

    def add(self, point_id: str, vector: dict[str, list[float]]) -> None:
        self.points.append(models.PointVectors(id=point_id, vector=vector))
        if len(self.points) >= BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if not self.points:
            return
        self.client.update_vectors(
            collection_name=self.collection_name,
            points=self.points,
        )
        self.points = []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest KernelPack chunks into Qdrant.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help='Absolute path to the package directory named "kernelpack".',
    )
    parser.add_argument("--papers", type=Path, help="Optional papers directory.")
    parser.add_argument(
        "--spaces",
        nargs="*",
        help="Named vector/sparse spaces to populate in this run.",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete stale code points absent from the current source tree.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source = args.source.expanduser()
    papers_dir = args.papers.expanduser() if args.papers is not None else None
    _validate_source(source)

    spaces = _requested_spaces(args.spaces, papers_dir is not None)
    _validate_spaces(spaces)

    qdrant = make_client()
    ensure_collections(qdrant)
    openai_client = OpenAI()
    registry = _build_embedder_registry(spaces, papers_dir is not None)

    symbol_table = build_symbol_table(source)
    stats = _RunStats()

    t0 = time.monotonic()
    _ingest_code(qdrant, openai_client, registry, source, spaces, symbol_table, stats)
    elapsed = time.monotonic() - t0

    reembedded = stats.coarse_chunks - stats.skipped_chunks
    cache_misses = reembedded - stats.summary_cache_hits
    print(
        f"\n{'─' * 37}\n"
        f" Ingest summary\n"
        f"{'─' * 37}\n"
        f" Total chunks seen:    {stats.coarse_chunks:>6}\n"
        f" Chunks skipped:       {stats.skipped_chunks:>6}  (unchanged)\n"
        f" Chunks re-embedded:   {reembedded:>6}  (new or changed)\n"
        f" Summary cache hits:   {stats.summary_cache_hits:>6}\n"
        f" Summary cache misses: {cache_misses:>6}\n"
        f" Elapsed:              {elapsed:>5.1f}s\n"
        f"{'─' * 37}"
    )

    if papers_dir is not None:
        _ingest_papers(qdrant, registry, papers_dir, spaces)
        if "math__qwen3" in spaces:
            _populate_math_vectors(qdrant, spaces)

    if args.prune:
        pruned = _prune_code_points(qdrant, source)
        print(f"Pruned code points: {pruned}")

    _print_invariant_report(qdrant, spaces, stats)


class _RunStats:
    def __init__(self) -> None:
        self.coarse_chunks = 0
        self.skipped_chunks = 0
        self.summary_cache_hits = 0
        self.coarse_with_cross_refs = 0


def _validate_source(source: Path) -> None:
    if not source.is_absolute():
        raise ValueError(f"--source must be absolute: {source}")
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if source.name != "kernelpack":
        raise ValueError(f'--source must be the directory named "kernelpack": {source}')


def _requested_spaces(raw_spaces: list[str] | None, has_papers: bool) -> tuple[str, ...]:
    if raw_spaces is not None:
        return tuple(dict.fromkeys(raw_spaces))

    spaces = list(DEFAULT_CODE_SPACES)
    if has_papers:
        spaces.extend(DEFAULT_PAPER_SPACES)
    return tuple(spaces)


def _validate_spaces(spaces: Iterable[str]) -> None:
    valid = set(COLLECTIONS_CONFIG[CODE_COLLECTION]["vectors"])
    valid.update(COLLECTIONS_CONFIG[CODE_COLLECTION]["sparse_vectors"])
    valid.update(COLLECTIONS_CONFIG[PAPERS_COLLECTION]["vectors"])
    valid.update(COLLECTIONS_CONFIG[PAPERS_COLLECTION]["sparse_vectors"])

    invalid = sorted(set(spaces) - valid)
    if invalid:
        raise ValueError(f"Unknown space(s): {', '.join(invalid)}")


def _build_embedder_registry(spaces: Iterable[str], has_papers: bool) -> EmbedderRegistry:
    registry = EmbedderRegistry()
    needed = {_model_name_for_space(space) for space in spaces}
    needed.discard(None)

    if "jinacode" in needed:
        from kernelpack_rag.embed.jinacode import JinaCodeEmbedder

        registry.register("jinacode", JinaCodeEmbedder())
    if "qwen3" in needed and has_papers:
        from kernelpack_rag.embed.qwen import QwenEmbedder

        registry.register("qwen3", QwenEmbedder())
    if "unixcoder" in needed:
        from kernelpack_rag.embed.unixcoder import UniXcoderEmbedder

        registry.register("unixcoder", UniXcoderEmbedder())

    return registry


def _model_name_for_space(space: str) -> str | None:
    if space.startswith("bm25_") or space == "summary__qwen3":
        return None
    if space == "math__qwen3":
        return "qwen3"
    if "__" not in space:
        return None
    suffix = space.rsplit("__", 1)[1]
    if suffix in {"jinacode", "qwen3", "unixcoder"}:
        return suffix
    return None


def _representation_key_for_space(space: str) -> RepresentationKey | None:
    prefix = space.split("__", 1)[0]
    if prefix == "ctx":
        return RepresentationKey.CTX
    if prefix == "code":
        return RepresentationKey.CODE
    if prefix == "codecom":
        return RepresentationKey.CODECOM
    if prefix == "com":
        return RepresentationKey.COM
    return None


def _load_existing_hashes(qdrant: QdrantClient) -> dict[str, str]:
    """Return {str(point_id): content_hash} for all existing coarse code points."""
    result: dict[str, str] = {}
    for point in _scroll_points(
        qdrant,
        CODE_COLLECTION,
        scroll_filter=_field_equals_filter("granularity", "coarse"),
        with_payload=True,
        with_vectors=False,
    ):
        h = (point.payload or {}).get("content_hash")
        if h:
            result[str(point.id)] = h
    return result


def _flush_pending(
    pending: list[tuple[str, dict, str, bool]],
    spaces: tuple[str, ...],
    registry: EmbedderRegistry,
    batcher: UpsertBatcher,
) -> None:
    """Embed all pending (id, payload, raw_text, is_coarse) tuples in batch and upsert."""
    if not pending:
        return

    all_vectors: list[dict] = [{} for _ in pending]

    for space in spaces:
        if space in ("summary__qwen3", "math__qwen3"):
            continue
        if space.startswith("bm25_"):
            continue
        key = _representation_key_for_space(space)
        if key is None:
            continue
        model_name = _model_name_for_space(space)
        if model_name is None:
            continue

        eligible: list[tuple[int, str]] = []
        for i, (_, payload, _, is_coarse) in enumerate(pending):
            if not is_coarse and space not in _FINE_ONLY_SPACES:
                continue
            text = build_representation(payload, key)
            if text is None:
                continue
            eligible.append((i, text))

        if not eligible:
            continue

        vecs = registry.get(model_name).embed_batch([t for _, t in eligible])
        for (i, _), vec in zip(eligible, vecs):
            all_vectors[i][space] = vec

    if "bm25_code" in spaces:
        for i, (_, _, raw_text, _) in enumerate(pending):
            all_vectors[i]["bm25_code"] = to_qdrant_sparse(raw_text)

    for (point_id, payload, _, _), vectors in zip(pending, all_vectors):
        batcher.add(models.PointStruct(id=point_id, vector=vectors, payload=payload))


def _ingest_code(
    qdrant: QdrantClient,
    openai_client: OpenAI,
    registry: EmbedderRegistry,
    source: Path,
    spaces: tuple[str, ...],
    symbol_table: dict[str, uuid.UUID],
    stats: _RunStats,
) -> None:
    batcher = UpsertBatcher(qdrant, CODE_COLLECTION)

    print("Scanning existing hashes...")
    existing_hashes = _load_existing_hashes(qdrant)

    print("Parsing source...")
    parsed: list[tuple[str, list[CoarseChunk]]] = []
    for py_file in _iter_source_py_files(source):
        source_text = py_file.read_text()
        parsed.append((source_text, chunk_file(py_file)))
    total_chunks = sum(len(file_chunks) for _, file_chunks in parsed)

    pending: list[tuple[str, dict, str, bool]] = []

    print("Ingesting chunks...")
    with tqdm(total=total_chunks, desc="chunks", unit="chunk") as pbar:
        for source_text, file_chunks in parsed:
            for chunk in file_chunks:
                coarse_id = get_coarse_uuid(chunk.source_file, chunk.qualname, chunk.module)
                chunk_hash = content_hash(chunk.text)
                if existing_hashes.get(str(coarse_id)) == chunk_hash:
                    stats.coarse_chunks += 1
                    stats.skipped_chunks += 1
                    pbar.update(1)
                    continue

                context_header = build_header(chunk, source_text, file_chunks)
                summary, _, cache_hit = _summarize_with_cache_status(chunk, openai_client)
                metadata = extract_metadata(chunk, symbol_table)
                payload = _coarse_payload(
                    chunk=chunk,
                    context_header=context_header,
                    summary=summary,
                    content_hash_value=chunk_hash,
                    metadata=metadata,
                )

                stats.coarse_chunks += 1
                if cache_hit:
                    stats.summary_cache_hits += 1
                if metadata.cross_ref_ids:
                    stats.coarse_with_cross_refs += 1

                pending.append((str(coarse_id), payload, chunk.text, True))

                for fine_chunk in fine_chunks(chunk):
                    fine_id = get_fine_uuid(
                        fine_chunk.source_file, fine_chunk.qualname, chunk.module, fine_chunk.window_idx
                    )
                    fine_payload = _fine_payload(
                        fine_chunk=fine_chunk,
                        parent_chunk=chunk,
                        context_header=context_header,
                        summary=summary,
                        content_hash_value=chunk_hash,
                        metadata=metadata,
                    )
                    pending.append((str(fine_id), fine_payload, fine_chunk.text, False))

                if len(pending) >= BATCH_SIZE:
                    _flush_pending(pending, spaces, registry, batcher)
                    pending.clear()

                pbar.update(1)

    print("Flushing remaining batch...")
    _flush_pending(pending, spaces, registry, batcher)
    batcher.flush()


def _summarize_with_cache_status(
    chunk: CoarseChunk,
    openai_client: OpenAI,
) -> tuple[str, str, bool]:
    h = content_hash(chunk.text)
    cache_path = SUMMARY_CACHE_DIR / f"{h}.txt"
    cache_hit = cache_path.exists() and cache_path.stat().st_size > 0
    summary, returned_hash = summarize_chunk(chunk, SUMMARY_CACHE_DIR, openai_client)
    return summary, returned_hash, cache_hit


def _coarse_payload(
    *,
    chunk: CoarseChunk,
    context_header: str,
    summary: str,
    content_hash_value: str,
    metadata,
) -> dict:
    return {
        "text": chunk.text,
        "context_header": context_header,
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
        "math_source_id": None,
        "content_hash": content_hash_value,
    }


def _fine_payload(
    *,
    fine_chunk,
    parent_chunk: CoarseChunk,
    context_header: str,
    summary: str,
    content_hash_value: str,
    metadata,
) -> dict:
    return {
        "text": fine_chunk.text,
        "context_header": context_header,
        "llm_summary": summary,
        "chunk_type": "fine",
        "module": parent_chunk.module,
        "parent_class": fine_chunk.parent_class,
        "function_name": metadata.function_name,
        "math_terms": metadata.math_terms,
        "cross_refs": metadata.cross_refs,
        "cross_ref_ids": [str(uid) for uid in metadata.cross_ref_ids],
        "has_numba": metadata.has_numba,
        "source_file": fine_chunk.source_file,
        "line_range": list(fine_chunk.line_range),
        "granularity": "fine",
        "parent_id": str(fine_chunk.parent_id),
        "math_source_id": None,
        "content_hash": content_hash_value,
    }


def _ingest_papers(
    qdrant: QdrantClient,
    registry: EmbedderRegistry,
    papers_dir: Path,
    spaces: tuple[str, ...],
) -> None:
    batcher = UpsertBatcher(qdrant, PAPERS_COLLECTION)
    paper_chunks = load_paper_chunks(papers_dir)

    for paper_chunk in paper_chunks:
        paper_id = uuid.uuid5(
            KP_NAMESPACE,
            f"paper:{paper_chunk.source_file}:{paper_chunk.section}:{content_hash(paper_chunk.text)[:16]}"
        )
        payload = _paper_payload(paper_chunk)
        batcher.add(
            models.PointStruct(
                id=str(paper_id),
                vector=_paper_vectors(paper_chunk, spaces, registry),
                payload=payload,
            )
        )

    batcher.flush()


def _paper_payload(paper_chunk: PaperChunk) -> dict:
    return {
        "text": paper_chunk.text,
        "math_terms": paper_chunk.math_terms,
        "section": paper_chunk.section,
        "equation_labels": paper_chunk.equation_labels,
        "source_file": paper_chunk.source_file,
        "content_hash": content_hash(paper_chunk.text),
    }


def _paper_vectors(
    paper_chunk: PaperChunk,
    spaces: tuple[str, ...],
    registry: EmbedderRegistry,
) -> dict:
    vectors: dict = {}
    if "paper__qwen3" in spaces:
        vectors["paper__qwen3"] = registry.get("qwen3").embed_batch([paper_chunk.text])[0]
    if "bm25_paper" in spaces:
        vectors["bm25_paper"] = to_qdrant_sparse(paper_chunk.text)
    return vectors


def _populate_math_vectors(
    qdrant: QdrantClient,
    spaces: tuple[str, ...],
) -> None:
    if "math__qwen3" not in spaces:
        return

    batcher = VectorUpdateBatcher(qdrant, CODE_COLLECTION)

    for point in _scroll_points(
        qdrant,
        CODE_COLLECTION,
        scroll_filter=_field_equals_filter("granularity", "coarse"),
        with_payload=True,
        with_vectors=False,
    ):
        payload = point.payload or {}
        math_terms = [term for term in payload.get("math_terms", []) if term]
        if not math_terms:
            continue

        paper_point = _find_paper_for_terms(qdrant, math_terms)
        if paper_point is None or not paper_point.payload:
            continue

        paper_vecs = qdrant.retrieve(
            collection_name=PAPERS_COLLECTION,
            ids=[paper_point.id],
            with_vectors=["paper__qwen3"],
        )
        if not paper_vecs or not paper_vecs[0].vector:
            continue
        paper_vec = paper_vecs[0].vector.get("paper__qwen3")
        if not paper_vec:
            continue
        batcher.add(str(point.id), {"math__qwen3": paper_vec})

        qdrant.set_payload(
            collection_name=CODE_COLLECTION,
            payload={"math_source_id": str(paper_point.id)},
            points=[point.id],
        )

    batcher.flush()


def _find_paper_for_terms(
    qdrant: QdrantClient,
    math_terms: list[str],
):
    # scroll limit=1 returns an arbitrary matching paper — correct only for a
    # single-paper corpus. Replace with vector search when corpus grows.
    points = list(
        _scroll_points(
            qdrant,
            PAPERS_COLLECTION,
            scroll_filter=_field_any_filter("math_terms", math_terms),
            with_payload=True,
            with_vectors=False,
            limit=1,
        )
    )
    return points[0] if points else None


def _prune_code_points(qdrant: QdrantClient, source: Path) -> int:
    current_ids = _current_code_ids(source)
    existing_ids = {
        str(point.id)
        for point in _scroll_points(
            qdrant,
            CODE_COLLECTION,
            with_payload=False,
            with_vectors=False,
        )
    }
    stale_ids = sorted(existing_ids - current_ids)
    if stale_ids:
        qdrant.delete(
            collection_name=CODE_COLLECTION,
            points_selector=models.PointIdsList(points=stale_ids),
        )
    return len(stale_ids)


def _iter_source_py_files(source: Path) -> list[Path]:
    """Yield all .py files: the kernelpack package plus its sibling tests/ dir."""
    files = list(source.rglob("*.py"))
    tests_dir = _find_tests_dir(source)
    if tests_dir is not None:
        files.extend(tests_dir.rglob("*.py"))
    return sorted(set(files))


def _find_tests_dir(source: Path) -> Path | None:
    """Return the tests/ directory adjacent to the package, if it exists."""
    for candidate in [
        source.parent / "tests",          # flat layout: <root>/kernelpack/../tests
        source.parent.parent / "tests",   # src layout:  <root>/src/kernelpack/../../tests
    ]:
        if candidate.is_dir():
            return candidate
    return None


def _current_code_ids(source: Path) -> set[str]:
    ids: set[str] = set()
    for py_file in _iter_source_py_files(source):
        for chunk in chunk_file(py_file):
            coarse_id = get_coarse_uuid(chunk.source_file, chunk.qualname, chunk.module)
            ids.add(str(coarse_id))
            for fine_chunk in fine_chunks(chunk):
                fine_id = get_fine_uuid(
                    fine_chunk.source_file, fine_chunk.qualname, chunk.module, fine_chunk.window_idx
                )
                ids.add(str(fine_id))
    return ids


def _print_invariant_report(
    qdrant: QdrantClient,
    spaces: tuple[str, ...],
    stats: _RunStats,
) -> None:
    code_points = list(
        _scroll_points(
            qdrant,
            CODE_COLLECTION,
            with_payload=True,
            with_vectors=["ctx__jinacode"],
        )
    )
    coarse_points = [
        point
        for point in code_points
        if (point.payload or {}).get("granularity") == "coarse"
    ]
    fine_points = [
        point for point in code_points if (point.payload or {}).get("granularity") == "fine"
    ]
    missing_primary = sum(
        1 for point in coarse_points if not _point_has_vector(point, "ctx__jinacode")
    )

    print("Invariant report")
    print(f"kernelpack_code coarse points: {len(coarse_points)}")
    print(f"kernelpack_code fine points: {len(fine_points)}")
    if missing_primary:
        print(f"ERROR: {missing_primary} coarse points missing primary vector")
    else:
        print("Coarse points missing ctx__jinacode vector: 0")

    com_spaces = [space for space in spaces if space.startswith("com__")]
    if com_spaces:
        for space in com_spaces:
            populated = sum(1 for point in coarse_points if _point_has_vector(point, space))
            rate = _rate(populated, len(coarse_points))
            print(f"{space} population rate: {populated}/{len(coarse_points)} ({rate:.2%})")
    else:
        print("com__* population rate: no com spaces requested")

    cross_ref_rate = _rate(stats.coarse_with_cross_refs, stats.coarse_chunks)
    print(
        "Cross-ref resolution rate: "
        f"{stats.coarse_with_cross_refs}/{stats.coarse_chunks} ({cross_ref_rate:.2%})"
    )

    summary_rate = _rate(stats.summary_cache_hits, stats.coarse_chunks)
    print(
        "Summary cache hit rate: "
        f"{stats.summary_cache_hits}/{stats.coarse_chunks} ({summary_rate:.2%})"
    )


def _point_has_vector(point, space: str) -> bool:
    vectors = getattr(point, "vector", None)
    if not isinstance(vectors, dict):
        return False
    vector = vectors.get(space)
    if vector is None:
        return False
    if hasattr(vector, "indices"):
        return bool(vector.indices)
    return bool(vector)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    main()
