"""Candidate rerankers for KernelPack RAG retrieval."""

from __future__ import annotations

import re

from kernelpack_rag.retrieve import Candidate


class NoopReranker:
    id = "NoopReranker"

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        return candidates


class CrossEncoderReranker:
    id = "CrossEncoderReranker"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        except Exception:
            self._model = None

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []

        if self._model is not None:
            pairs = [(query, _candidate_text(candidate)) for candidate in candidates]
            raw_scores = self._model.predict(pairs)
            scores = [float(score) for score in raw_scores]
        else:
            scores = [
                _lexical_fallback_score(query, _candidate_text(candidate))
                for candidate in candidates
            ]

        for candidate, score in zip(candidates, scores):
            candidate.leg_scores["rerank"] = float(score)

        reranked = sorted(
            candidates,
            key=lambda candidate: candidate.leg_scores["rerank"],
            reverse=True,
        )
        for rank, candidate in enumerate(reranked):
            candidate.fused_rank = rank
        return reranked


def _candidate_text(candidate: Candidate) -> str:
    payload = candidate.payload or {}
    parts = [
        payload.get("llm_summary"),
        payload.get("context_header"),
        payload.get("text"),
    ]
    return "\n".join(str(part) for part in parts if part)


def _lexical_fallback_score(query: str, text: str) -> float:
    query_terms = set(_terms(query))
    if not query_terms:
        return 0.0
    text_terms = set(_terms(text))
    overlap = len(query_terms & text_terms)
    return float(overlap / len(query_terms))


def _terms(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


__all__ = ["NoopReranker", "CrossEncoderReranker"]
