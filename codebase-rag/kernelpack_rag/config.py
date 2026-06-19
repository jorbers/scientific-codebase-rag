"""Shared configuration for Qdrant collection schemas."""

from __future__ import annotations

import os

from qdrant_client import QdrantClient


CODE_COLLECTION = "kernelpack_code"
PAPERS_COLLECTION = "kernelpack_papers"


COLLECTIONS_CONFIG = {
    "kernelpack_code": {
        "vectors": {
            "ctx__jinacode": {"size": 896, "distance": "Cosine"},
            "ctx__qwen3": {"size": 1024, "distance": "Cosine"},
            "ctx__unixcoder": {"size": 768, "distance": "Cosine"},
            "code__jinacode": {"size": 896, "distance": "Cosine"},
            "code__qwen3": {"size": 1024, "distance": "Cosine"},
            "code__unixcoder": {"size": 768, "distance": "Cosine"},
            "codecom__jinacode": {"size": 896, "distance": "Cosine"},
            "codecom__qwen3": {"size": 1024, "distance": "Cosine"},
            "codecom__unixcoder": {"size": 768, "distance": "Cosine"},
            "com__jinacode": {"size": 896, "distance": "Cosine"},
            "com__qwen3": {"size": 1024, "distance": "Cosine"},
            "com__unixcoder": {"size": 768, "distance": "Cosine"},
            "math__qwen3": {"size": 1024, "distance": "Cosine"},
            "summary__qwen3": {"size": 1024, "distance": "Cosine"},
        },
        "sparse_vectors": {
            "bm25_code": {"modifier": "idf"},
        },
    },
    "kernelpack_papers": {
        "vectors": {
            "paper__qwen3": {"size": 1024, "distance": "Cosine"},
        },
        "sparse_vectors": {
            "bm25_paper": {"modifier": "idf"},
        },
    },
}


def make_client() -> QdrantClient:
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port)


__all__ = ["CODE_COLLECTION", "COLLECTIONS_CONFIG", "make_client", "PAPERS_COLLECTION"]
