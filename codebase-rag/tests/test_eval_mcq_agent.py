"""Unit tests for eval_mcq_agent: format_question, extract_letter, _call_handler."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from eval_mcq_agent import _call_handler, extract_letter, format_question
from kernelpack_rag.retrieve import Candidate


# ---------------------------------------------------------------------------
# format_question
# ---------------------------------------------------------------------------

SAMPLE_Q = {
    "id": "kp_mcq_001",
    "difficulty": "easy",
    "category": "package_structure",
    "question": "Which set of top-level subpackages is exported by `kernelpack`?",
    "options": [
        "geometry, nodes, domain, poly, rbffd, solvers",
        "divfree, domain, geometry, nodes, poly, rbffd, solvers",
        "geometry, domain, poly, solvers",
        "divfree, geometry, poly, tests",
    ],
    "answer": "B",
    "rationale": "The top-level package exports seven subpackages, including `divfree`.",
    "source_refs": ["src/kernelpack/__init__.py:1-3"],
}


def test_format_question_labels():
    text = format_question(SAMPLE_Q)
    assert "A. geometry, nodes, domain" in text
    assert "B. divfree, domain, geometry" in text
    assert "C. geometry, domain, poly, solvers" in text
    assert "D. divfree, geometry, poly, tests" in text


def test_format_question_contains_question_text():
    text = format_question(SAMPLE_Q)
    assert "Which set of top-level subpackages" in text


def test_format_question_no_source_refs():
    text = format_question(SAMPLE_Q)
    assert "source_refs" not in text
    assert "__init__.py" not in text


def test_format_question_order_matches_options():
    q = {**SAMPLE_Q, "options": ["first", "second", "third", "fourth"]}
    text = format_question(q)
    labeled = [l for l in text.splitlines() if l.startswith(("A.", "B.", "C.", "D."))]
    assert labeled[0] == "A. first"
    assert labeled[1] == "B. second"
    assert labeled[2] == "C. third"
    assert labeled[3] == "D. fourth"


# ---------------------------------------------------------------------------
# extract_letter
# ---------------------------------------------------------------------------


def test_extract_letter_valid():
    letter, reasoning = extract_letter('{"answer": "B", "reasoning": "chunk showed..."}')
    assert letter == "B"
    assert reasoning == "chunk showed..."


def test_extract_letter_lowercase():
    letter, _ = extract_letter('{"answer": "c", "reasoning": "..."}')
    assert letter == "C"


def test_extract_letter_strips_json_fence():
    text = '```json\n{"answer": "A", "reasoning": "found it"}\n```'
    letter, reasoning = extract_letter(text)
    assert letter == "A"
    assert reasoning == "found it"


def test_extract_letter_strips_plain_fence():
    text = '```\n{"answer": "D", "reasoning": "ok"}\n```'
    letter, _ = extract_letter(text)
    assert letter == "D"


def test_extract_letter_invalid_letter():
    with pytest.raises((ValueError, KeyError)):
        extract_letter('{"answer": "E", "reasoning": "..."}')


def test_extract_letter_bad_json():
    with pytest.raises((json.JSONDecodeError, ValueError)):
        extract_letter("not json at all")


def test_extract_letter_missing_answer_key():
    with pytest.raises((KeyError, ValueError)):
        extract_letter('{"reasoning": "hmm"}')


# ---------------------------------------------------------------------------
# _call_handler
# ---------------------------------------------------------------------------


def _make_candidate(function_name: str, score: float, rank: int) -> Candidate:
    return Candidate(
        point_id=f"pt-{rank}",
        payload={
            "function_name": function_name,
            "module": "kernelpack.geometry.core",
            "chunk_type": "function",
            "text": f"def {function_name}(): ...",
            "llm_summary": "summary",
        },
        leg_scores={"dense": score, "sparse": score},
        fused_rank=rank,
        fused_score=score,
    )


def _make_tool_call(query: str, module_filter: str | None = None) -> SimpleNamespace:
    args: dict = {"query": query}
    if module_filter is not None:
        args["module_filter"] = module_filter
    return SimpleNamespace(
        id="tc-001",
        function=SimpleNamespace(
            name="retrieve_code",
            arguments=json.dumps(args),
        ),
    )


@patch("eval_mcq_agent.hybrid")
def test_call_handler_passes_query_and_collection(mock_hybrid):
    mock_hybrid.return_value = [_make_candidate("foo", 0.9, 0), _make_candidate("bar", 0.5, 1)]
    tc = _make_tool_call("kernelpack subpackages")
    _call_handler(tc, client=MagicMock(), embedder=MagicMock(), log_path=Path("logs/eval_agent.jsonl"))
    kwargs = mock_hybrid.call_args
    assert kwargs[0][0] == "kernelpack subpackages"
    assert kwargs[1]["collection"] == "kernelpack_code"
    assert kwargs[1]["query_id"] is not None
    assert kwargs[1]["log_path"] == Path("logs/eval_agent.jsonl")


@patch("eval_mcq_agent.hybrid")
def test_call_handler_chunks_returned(mock_hybrid):
    mock_hybrid.return_value = [_make_candidate("foo", 0.9, 0), _make_candidate("bar", 0.5, 1)]
    tc = _make_tool_call("some query")
    content, summary = _call_handler(
        tc, client=MagicMock(), embedder=MagicMock(), log_path=Path("logs/eval_agent.jsonl")
    )
    assert summary["chunks_returned"] == 2
    assert summary["query"] == "some query"
    assert len(json.loads(content)) == 2


@patch("eval_mcq_agent.hybrid")
def test_call_handler_top_chunk_qualname(mock_hybrid):
    mock_hybrid.return_value = [_make_candidate("make_grid", 0.91, 0)]
    tc = _make_tool_call("grid setup")
    _, summary = _call_handler(
        tc, client=MagicMock(), embedder=MagicMock(), log_path=Path("logs/eval_agent.jsonl")
    )
    assert summary["top_chunk_qualname"] == "make_grid"
    assert summary["top_chunk_score"] == 0.91


@patch("eval_mcq_agent.hybrid")
def test_call_handler_module_filter_applied(mock_hybrid):
    mock_hybrid.return_value = []
    tc = _make_tool_call("poisson solver", module_filter="kernelpack.solvers.poisson")
    _call_handler(tc, client=MagicMock(), embedder=MagicMock(), log_path=Path("logs/eval_agent.jsonl"))
    assert mock_hybrid.call_args[1]["query_filter"] is not None


@patch("eval_mcq_agent.hybrid")
def test_call_handler_no_module_filter(mock_hybrid):
    mock_hybrid.return_value = []
    tc = _make_tool_call("any query")
    _call_handler(tc, client=MagicMock(), embedder=MagicMock(), log_path=Path("logs/eval_agent.jsonl"))
    assert mock_hybrid.call_args[1]["query_filter"] is None


@patch("eval_mcq_agent.hybrid")
def test_call_handler_empty_results(mock_hybrid):
    mock_hybrid.return_value = []
    tc = _make_tool_call("nothing found")
    content, summary = _call_handler(
        tc, client=MagicMock(), embedder=MagicMock(), log_path=Path("logs/eval_agent.jsonl")
    )
    assert summary["chunks_returned"] == 0
    assert summary["top_chunk_qualname"] is None
    assert summary["top_chunk_score"] is None
    assert json.loads(content) == []
