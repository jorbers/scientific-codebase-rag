"""Unit tests for qlog.py (Step 15).

qlog.py is the pipeline's observability layer. These tests verify:
- Events are written as valid JSONL and can be read back
- Append-only behavior: second write does not overwrite first
- fold() correctly merges all events for a query_id into one record
- Oracle mode is preserved faithfully (injected_ids non-empty, candidates empty)
- The not_in_retrieved / not_in_package split is maintained as separate fields

No Qdrant required. All tests use tmp_path and in-memory data.

The identifier attribution split (not_in_retrieved vs not_in_package) is the
core of the failure attribution system — the reason this pipeline is novel.
Tests for it are not optional.
"""

import json
import uuid
from pathlib import Path

import pytest

from kernelpack_rag.qlog import fold, write_event


# ── helpers ────────────────────────────────────────────────────────────────────

def _query_id() -> str:
    return str(uuid.uuid4())


def _retrieval_event(query_id: str, candidates=None, injected_ids=None) -> dict:
    return {
        "query_id": query_id,
        "event": "retrieval",
        "payload": {
            "query_text": "normalize rows of a matrix",
            "plan": "hybrid",
            "spaces": ["ctx__jinacode", "bm25_code"],
            "fusion": "rrf",
            "filters": [],
            "reranker_id": "NoopReranker",
            "candidates": candidates if candidates is not None else [
                {
                    "point_id": "abc123",
                    "leg_scores": {"dense": 0.91, "sparse": 12.3},
                    "fused_rank": 0,
                    "fused_score": 0.032,
                },
                {
                    "point_id": "def456",
                    "leg_scores": {"dense": 0.87, "sparse": 9.1},
                    "fused_rank": 1,
                    "fused_score": 0.028,
                },
            ],
            "injected_ids": injected_ids if injected_ids is not None else [],
        },
    }


def _generation_event(
    query_id: str,
    not_in_retrieved=None,
    not_in_package=None,
) -> dict:
    return {
        "query_id": query_id,
        "event": "generation",
        "payload": {
            "model": "gpt-4o",
            "response_text": "Use normalize_rows(X) to scale each row to unit norm.",
            "referenced_ids": ["abc123"],
            "unreferenced_ids": [],
            "identifiers_not_in_retrieved": not_in_retrieved or [],
            "identifiers_not_in_package": not_in_package or [],
        },
    }


def _execution_event(query_id: str, result: str = "CorrectOutput") -> dict:
    return {
        "query_id": query_id,
        "event": "execution",
        "payload": {
            "result": result,
            "detail": "",
        },
    }


# ── write_event ────────────────────────────────────────────────────────────────

class TestWriteEvent:
    def test_creates_file_if_not_exists(self, tmp_path):
        log = tmp_path / "subdir" / "run.jsonl"
        assert not log.exists()
        write_event(log, _retrieval_event(_query_id()))
        assert log.exists()

    def test_written_line_is_valid_json(self, tmp_path):
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _retrieval_event(qid))
        lines = log.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["query_id"] == qid
        assert record["event"] == "retrieval"

    def test_append_does_not_overwrite(self, tmp_path):
        """Two write_event calls must produce two lines, not one."""
        log = tmp_path / "run.jsonl"
        qid1, qid2 = _query_id(), _query_id()
        write_event(log, _retrieval_event(qid1))
        write_event(log, _generation_event(qid2))
        lines = log.read_text().strip().split("\n")
        assert len(lines) == 2
        ids = [json.loads(l)["query_id"] for l in lines]
        assert qid1 in ids and qid2 in ids

    def test_ts_field_present(self, tmp_path):
        """Every event must carry a timestamp — required for log ordering."""
        log = tmp_path / "run.jsonl"
        write_event(log, _retrieval_event(_query_id()))
        record = json.loads(log.read_text().strip())
        assert "ts" in record, "write_event must add a 'ts' timestamp field"

    def test_payload_survives_roundtrip(self, tmp_path):
        """Payload fields must be preserved exactly — no silent truncation."""
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        event = _retrieval_event(qid)
        write_event(log, event)
        record = json.loads(log.read_text().strip())
        assert record["payload"]["query_text"] == "normalize rows of a matrix"
        assert record["payload"]["plan"] == "hybrid"
        assert len(record["payload"]["candidates"]) == 2
        assert record["payload"]["candidates"][0]["point_id"] == "abc123"
        assert record["payload"]["candidates"][0]["leg_scores"]["dense"] == 0.91


# ── fold ───────────────────────────────────────────────────────────────────────

class TestFold:
    def test_returns_none_for_unknown_query_id(self, tmp_path):
        log = tmp_path / "run.jsonl"
        write_event(log, _retrieval_event(_query_id()))
        assert fold(log, _query_id()) is None

    def test_single_retrieval_event_folds(self, tmp_path):
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _retrieval_event(qid))
        record = fold(log, qid)
        assert record is not None
        assert "retrieval" in record

    def test_fold_merges_all_three_event_types(self, tmp_path):
        """fold must merge retrieval + generation + execution into one record."""
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _retrieval_event(qid))
        write_event(log, _generation_event(qid))
        write_event(log, _execution_event(qid))

        record = fold(log, qid)
        assert record is not None
        assert "retrieval" in record, "fold missing retrieval key"
        assert "generation" in record, "fold missing generation key"
        assert "execution" in record, "fold missing execution key"

    def test_fold_preserves_candidates(self, tmp_path):
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _retrieval_event(qid))
        record = fold(log, qid)
        candidates = record["retrieval"]["candidates"]
        assert len(candidates) == 2
        assert candidates[0]["point_id"] == "abc123"
        assert candidates[0]["fused_rank"] == 0

    def test_fold_ignores_other_query_ids(self, tmp_path):
        """fold must not leak events from other queries into the result."""
        log = tmp_path / "run.jsonl"
        qid_target = _query_id()
        qid_noise = _query_id()
        write_event(log, _retrieval_event(qid_noise))
        write_event(log, _retrieval_event(qid_target))
        write_event(log, _generation_event(qid_noise))

        record = fold(log, qid_target)
        assert "generation" not in record, (
            "fold included generation event from a different query_id"
        )

    def test_fold_handles_partial_events(self, tmp_path):
        """fold must work even if generation/execution events haven't been written yet."""
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _retrieval_event(qid))
        record = fold(log, qid)
        assert "retrieval" in record
        assert "generation" not in record
        assert "execution" not in record

    def test_fold_execution_result_preserved(self, tmp_path):
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _retrieval_event(qid))
        write_event(log, _generation_event(qid))
        write_event(log, _execution_event(qid, result="WrongSignature"))
        record = fold(log, qid)
        assert record["execution"]["result"] == "WrongSignature"


# ── oracle mode ────────────────────────────────────────────────────────────────

class TestOracleMode:
    def test_injected_ids_preserved(self, tmp_path):
        """
        Oracle mode bypasses Qdrant and injects known chunk IDs directly.
        The injected IDs must survive the write/fold roundtrip unchanged —
        the eval harness reads them to attribute failures.
        """
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        injected = ["point-id-001", "point-id-002", "point-id-003"]
        event = _retrieval_event(qid, candidates=[], injected_ids=injected)
        write_event(log, event)
        record = fold(log, qid)
        assert record["retrieval"]["injected_ids"] == injected

    def test_oracle_mode_candidates_empty(self, tmp_path):
        """In oracle mode, candidates must be empty (Qdrant was not queried)."""
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _retrieval_event(qid, candidates=[], injected_ids=["id-1"]))
        record = fold(log, qid)
        assert record["retrieval"]["candidates"] == []
        assert len(record["retrieval"]["injected_ids"]) > 0


# ── identifier attribution ─────────────────────────────────────────────────────

class TestIdentifierAttribution:
    def test_not_in_retrieved_and_not_in_package_are_separate_fields(self, tmp_path):
        """
        These two fields are the core of the failure attribution system.
        not_in_retrieved  = hallucinated function that EXISTS in the package
                            but was not retrieved → retrieval failure
        not_in_package    = hallucinated function that does NOT exist at all
                            → true hallucination

        They must be stored as separate fields. Collapsing them loses the
        distinction between retrieval failures and generation failures.
        """
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(
            log,
            _generation_event(
                qid,
                not_in_retrieved=["normalize_rows"],   # exists in package, missed by retrieval
                not_in_package=["fake_solver"],         # does not exist — hallucination
            ),
        )
        record = fold(log, qid)
        gen = record["generation"]

        assert "identifiers_not_in_retrieved" in gen
        assert "identifiers_not_in_package" in gen
        assert gen["identifiers_not_in_retrieved"] == ["normalize_rows"]
        assert gen["identifiers_not_in_package"] == ["fake_solver"]
        assert gen["identifiers_not_in_retrieved"] != gen["identifiers_not_in_package"], (
            "The two failure classes must remain distinct. "
            "Merging them makes failure attribution impossible."
        )

    def test_both_fields_empty_for_correct_generation(self, tmp_path):
        """A correct generation produces no attribution failures."""
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(log, _generation_event(qid, not_in_retrieved=[], not_in_package=[]))
        record = fold(log, qid)
        gen = record["generation"]
        assert gen["identifiers_not_in_retrieved"] == []
        assert gen["identifiers_not_in_package"] == []

    def test_retrieval_miss_does_not_count_as_hallucination(self, tmp_path):
        """
        A function that exists in the package but wasn't retrieved is a
        retrieval miss, not a hallucination. It must appear only in
        not_in_retrieved, never in not_in_package.
        """
        log = tmp_path / "run.jsonl"
        qid = _query_id()
        write_event(
            log,
            _generation_event(
                qid,
                not_in_retrieved=["build_stencil"],
                not_in_package=[],
            ),
        )
        record = fold(log, qid)
        gen = record["generation"]
        assert "build_stencil" in gen["identifiers_not_in_retrieved"]
        assert "build_stencil" not in gen["identifiers_not_in_package"]