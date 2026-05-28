"""Migrate the exported ChromaDB v3 index into a Qdrant collection.

The script reads ``experiments/chroma_export.pkl`` by default. That pickle is
expected to be the dict returned by ``chroma_col.get(include=[
"documents", "metadatas", "embeddings"])`` in the baseline notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


COLLECTION_NAME = "kernelpack-v3"
BATCH_SIZE = 100

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
DEFAULT_EXPORT_PATH = EXPERIMENTS_DIR / "chroma_export.pkl"
DEFAULT_STORAGE_PATH = PROJECT_ROOT / "qdrant_storage"


@dataclass(frozen=True)
class ChromaExport:
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, Any]]
    embeddings: Any

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def vector_dim(self) -> int:
        return len(self.embeddings[0])


def stable_point_id(string_id: str) -> int:
    """Return a deterministic Qdrant integer point ID for a ChromaDB string ID."""
    digest = hashlib.sha256(string_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and other simple objects into Qdrant payload values."""
    if hasattr(value, "item"):
        return value.item()
    return value


def load_chroma_export(export_path: Path) -> ChromaExport:
    with export_path.open("rb") as f:
        raw = pickle.load(f)

    required_keys = {"ids", "documents", "metadatas", "embeddings"}
    missing = sorted(required_keys - set(raw))
    if missing:
        raise KeyError(f"{export_path} is missing required keys: {missing}")

    export = ChromaExport(
        ids=list(raw["ids"]),
        documents=list(raw["documents"]),
        metadatas=list(raw["metadatas"]),
        embeddings=raw["embeddings"],
    )

    lengths = {
        "ids": len(export.ids),
        "documents": len(export.documents),
        "metadatas": len(export.metadatas),
        "embeddings": len(export.embeddings),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Chroma export has mismatched lengths: {lengths}")
    if export.count == 0:
        raise ValueError(f"Chroma export is empty: {export_path}")

    return export


def make_qdrant_client(
    *,
    storage_path: Path | None = DEFAULT_STORAGE_PATH,
    host: str | None = None,
    port: int = 6333,
) -> QdrantClient:
    """Create a Qdrant client for either local file storage or a running server."""
    if host:
        return QdrantClient(host=host, port=port)

    if storage_path is None:
        raise ValueError("storage_path is required when host is not provided")

    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def recreate_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    vector_dim: int,
    recreate: bool,
) -> None:
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )


def build_points(export: ChromaExport) -> list[PointStruct]:
    points: list[PointStruct] = []
    for document, metadata, embedding, source_id in zip(
        export.documents,
        export.metadatas,
        export.embeddings,
        export.ids,
    ):
        payload = {key: _json_safe(value) for key, value in metadata.items()}
        payload["text"] = document
        payload["original_id"] = source_id

        points.append(
            PointStruct(
                id=stable_point_id(source_id),
                vector=[float(value) for value in embedding],
                payload=payload,
            )
        )
    return points


def migrate_to_qdrant(
    *,
    export_path: Path = DEFAULT_EXPORT_PATH,
    storage_path: Path | None = DEFAULT_STORAGE_PATH,
    host: str | None = None,
    port: int = 6333,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
    recreate: bool = True,
) -> int:
    export = load_chroma_export(export_path)
    client = make_qdrant_client(storage_path=storage_path, host=host, port=port)

    try:
        recreate_collection(
            client,
            collection_name=collection_name,
            vector_dim=export.vector_dim,
            recreate=recreate,
        )

        points = build_points(export)
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
            print(f"Upserted {min(start + batch_size, len(points))}/{len(points)}")

        qdrant_count = client.get_collection(collection_name).points_count
        if qdrant_count != export.count:
            raise RuntimeError(
                f"Count mismatch after migration: Chroma export={export.count}, "
                f"Qdrant={qdrant_count}"
            )

        print(
            f"Migration complete: {qdrant_count} points in "
            f"Qdrant collection {collection_name!r}"
        )
        return qdrant_count
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate experiments/chroma_export.pkl into Qdrant."
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help="Path to the ChromaDB pickle export.",
    )
    parser.add_argument(
        "--storage-path",
        type=Path,
        default=DEFAULT_STORAGE_PATH,
        help="Qdrant local storage path. Ignored when --host is provided.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Qdrant server host. If omitted, local file storage is used.",
    )
    parser.add_argument("--port", type=int, default=6333)
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Upsert into an existing collection instead of deleting/recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    migrate_to_qdrant(
        export_path=args.export_path,
        storage_path=args.storage_path,
        host=args.host,
        port=args.port,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
        recreate=not args.no_recreate,
    )


if __name__ == "__main__":
    main()
