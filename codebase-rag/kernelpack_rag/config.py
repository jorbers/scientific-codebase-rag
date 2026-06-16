"""Shared configuration for Qdrant collection schemas."""

from __future__ import annotations


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


__all__ = ["COLLECTIONS_CONFIG"]
