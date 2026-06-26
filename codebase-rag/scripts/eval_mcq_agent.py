#!/usr/bin/env python3
"""Agent-based MCQ evaluation for KernelPack RAG.

Runs each question through gpt-4o with a retrieve_code tool, one fresh
conversation per question. Measures answer accuracy after forced retrieval.

Usage:
    python eval_mcq_agent.py --mcq path/to/mcq.json
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path before importing project modules.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os
import re
import uuid

import openai
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

from kernelpack_rag.config import CODE_COLLECTION, make_client
from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
from kernelpack_rag.qdrant_utils import _field_equals_filter
from kernelpack_rag.retrieve import Candidate, hybrid

LOG_PATH = REPO_ROOT / "logs" / "eval_agent.jsonl"
OUTPUT_DIR = REPO_ROOT / "eval" / "results"
OUTPUT_JSON = OUTPUT_DIR / "mcq_agent_baseline.json"
OUTPUT_MD = OUTPUT_DIR / "mcq_agent_baseline.md"

SYSTEM_PROMPT = """\
You are being evaluated on a multiple choice question about the KernelPack library.

QUERY FORMATION RULES — read carefully before issuing any retrieve_code call:
- Every query MUST contain at least 2-3 terms that combine the identifier with
  context: the module name, what the identifier does, or related concepts.
- NEVER issue a single-word or single-identifier query. Single-term queries
  return noise and waste your budget.

  Good: "DomainDescriptor kernelpack.domain exported class"
  Good: "kernelpack.domain module exports subpackage"
  Bad:  "DomainDescriptor"
  Bad:  "exports"

- For each answer option, query by name WITH context before ruling it out.
  Example: if option B is "register_kernel", query
  "register_kernel kernelpack function purpose" — not just "register_kernel".

RETRIEVAL REQUIREMENTS:
- You MUST call retrieve_code at least once before answering.
- You MUST try at least 5 distinct multi-term queries before flagging
  insufficient_context. Each retry must use different terms or angles
  (try the module path, the identifier + what it does, related concepts,
  neighboring identifiers mentioned in the question).
- Answer ONLY from what retrieve_code returns. Do not use prior knowledge
  about KernelPack.

OUTPUT FORMAT:
- Respond only with JSON: {"answer": "B", "reasoning": "..."}
  Your reasoning MUST reference specific content from retrieved chunks
  (e.g. function names, docstring text, exact line content).
- If after at least 5 distinct multi-term queries you still cannot find
  specific evidence, output:
  {"answer": "X", "reasoning": "could not find specific evidence because..."}\
"""

RETRIEVE_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve_code",
        "description": "Search the KernelPack code index and return matching code chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "module_filter": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}

_LETTERS = ("A", "B", "C", "D")
_VALID_ANSWERS = (*_LETTERS, "X")


def format_question(q: dict) -> str:
    lines = [q["question"], ""]
    for letter, option in zip(_LETTERS, q["options"]):
        lines.append(f"{letter}. {option}")
    return "\n".join(lines)


def extract_letter(text: str) -> tuple[str, str]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    parsed = json.loads(text)
    answer = str(parsed["answer"]).strip().upper()
    if answer not in _VALID_ANSWERS:
        raise ValueError(f"answer {answer!r} not in A-D or X")
    return answer, str(parsed.get("reasoning", ""))


def _shorten_source(path: str) -> str:
    marker = "kernelpack-python/"
    idx = path.find(marker)
    return path[idx:] if idx != -1 else path


def _qualname(candidate: Candidate) -> str:
    p = candidate.payload
    source = _shorten_source(str(p.get("source_file") or ""))
    fn = str(p.get("function_name") or "")
    if source:
        return f"{source}:{fn}" if fn else source
    if fn:
        return fn
    module = str(p.get("module") or "")
    chunk_type = str(p.get("chunk_type") or "")
    return f"{module}::{chunk_type}" if module else chunk_type


def _chunk_to_llm_dict(candidate: Candidate) -> dict:
    p = candidate.payload
    source = _shorten_source(str(p.get("source_file") or ""))
    fn = str(p.get("function_name") or "")
    return {
        "source": f"{source}:{fn}" if fn else source,
        "function_name": fn,
        "chunk_type": str(p.get("chunk_type") or ""),
        "text": str(p.get("text") or ""),
        "llm_summary": str(p.get("llm_summary") or ""),
    }


def _call_handler(
    tc,
    *,
    client: QdrantClient,
    embedder,
    log_path: Path,
) -> tuple[str, dict]:
    args = json.loads(tc.function.arguments)
    query = args["query"]
    module_filter = args.get("module_filter")
    query_filter = _field_equals_filter("module", module_filter) if module_filter else None

    candidates = hybrid(
        query,
        client=client,
        collection=CODE_COLLECTION,
        embedder=embedder,
        k=10,
        query_filter=query_filter,
        query_id=str(uuid.uuid4()),
        log_path=log_path,
    )
    chunks = [_chunk_to_llm_dict(c) for c in candidates]
    top = candidates[0] if candidates else None
    call_summary = {
        "query": query,
        "chunks_returned": len(chunks),
        "top_chunk_qualname": _qualname(top) if top else None,
        "top_chunk_score": round(top.fused_score, 4) if top else None,
    }
    return json.dumps(chunks), call_summary


def _run_question(
    q: dict,
    *,
    oai: openai.OpenAI,
    client: QdrantClient,
    embedder,
    log_path: Path,
) -> dict:
    history: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_question(q)},
    ]
    retrieval_calls: list[dict] = []
    flags: list[str] = []

    resp = oai.chat.completions.create(
        model="o4-mini",
        reasoning_effort="medium",
        messages=history,
        tools=[RETRIEVE_CODE_TOOL],
        tool_choice="required",
    )

    while resp.choices[0].finish_reason == "tool_calls":
        msg = resp.choices[0].message
        history.append(msg)
        for tc in msg.tool_calls:
            content, call_summary = _call_handler(
                tc, client=client, embedder=embedder, log_path=log_path
            )
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })
            retrieval_calls.append(call_summary)
        resp = oai.chat.completions.create(
            model="o4-mini",
            reasoning_effort="medium",
            messages=history,
            tools=[RETRIEVE_CODE_TOOL],
        )

    if not retrieval_calls:
        flags.append("no_retrieval")

    final_text = resp.choices[0].message.content or ""
    predicted: str | None = None
    reasoning: str | None = None
    try:
        predicted, reasoning = extract_letter(final_text)
    except Exception:
        flags.append("parse_error")

    if predicted == "X":
        flags.append("insufficient_context")
        predicted = None

    correct = q["answer"]
    passed: bool | None = (predicted == correct) if predicted is not None else None

    return {
        "mcq_id": q["id"],
        "question": q["question"],
        "options": {letter: text for letter, text in zip(_LETTERS, q["options"])},
        "correct_answer": correct,
        "predicted_answer": predicted,
        "reasoning": reasoning,
        "pass": passed,
        "retrieval_calls": retrieval_calls,
        "flags": flags,
        "_difficulty": q["difficulty"],
        "_category": q["category"],
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _pass_rate(results: list[dict]) -> tuple[int, int]:
    scored = [r for r in results if not r["flags"]]
    return sum(1 for r in scored if r["pass"]), len(scored)


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "0/0"


def _build_md(results: list[dict]) -> str:
    lines: list[str] = []
    p, d = _pass_rate(results)

    lines += ["# MCQ Agent Baseline Results", "", f"**Overall pass rate:** {_pct(p, d)}", ""]

    lines += ["## Pass rate by difficulty", ""]
    for diff in ("easy", "medium", "hard"):
        scored = [r for r in results if r["_difficulty"] == diff and not r["flags"]]
        p_sub = sum(1 for r in scored if r["pass"])
        lines.append(f"- **{diff}**: {_pct(p_sub, len(scored))}")
    lines.append("")

    lines += ["## Pass rate by category", ""]
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["_category"], []).append(r)
    for cat, rs in sorted(categories.items()):
        scored = [r for r in rs if not r["flags"]]
        p_cat = sum(1 for r in scored if r["pass"])
        lines.append(f"- **{cat}**: {_pct(p_cat, len(scored))}")
    lines.append("")

    parse_errors = sum(1 for r in results if "parse_error" in r["flags"])
    no_retrievals = sum(1 for r in results if "no_retrieval" in r["flags"])
    insufficient = sum(1 for r in results if "insufficient_context" in r["flags"])
    lines += [
        "## Flags",
        "",
        f"- `parse_error`: {parse_errors}",
        f"- `no_retrieval`: {no_retrievals}",
        f"- `insufficient_context`: {insufficient}",
        "",
    ]

    lines += [
        "## Per-question results",
        "",
        "| id | difficulty | pass | retrieval calls | flags |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r["pass"] is None:
            pass_str = "—"
        elif r["pass"]:
            pass_str = "pass"
        else:
            pass_str = "fail"
        flags_str = ", ".join(r["flags"]) if r["flags"] else ""
        lines.append(
            f"| {r['mcq_id']} | {r['_difficulty']} | {pass_str} |"
            f" {len(r['retrieval_calls'])} | {flags_str} |"
        )
    lines.append("")

    failed = [r for r in results if r["pass"] is False]
    if failed:
        lines += ["## Failed questions", ""]
        for r in failed:
            top = r["retrieval_calls"][0] if r["retrieval_calls"] else None
            top_str = (
                f"`{top['top_chunk_qualname']}` (score {top['top_chunk_score']})"
                if top
                else "none"
            )
            lines += [
                f"### {r['mcq_id']}",
                "",
                f"**Question:** {r['question']}",
                f"**Correct:** {r['correct_answer']}  **Predicted:** {r['predicted_answer']}",
                f"**Reasoning:** {r['reasoning']}",
                f"**Top retrieved chunk:** {top_str}",
                "",
            ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcq", type=Path, required=True, metavar="PATH")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Run only the first N questions (default: all).")
    args = parser.parse_args(argv)

    if not args.mcq.exists():
        sys.exit(f"error: MCQ file not found: {args.mcq}")

    mcqs: list[dict] = json.loads(args.mcq.read_text())
    if args.limit is not None:
        mcqs = mcqs[: args.limit]
    print(f"Loaded {len(mcqs)} questions.")

    print("Loading embedder...")
    embedder = JinaCodeEmbedder()
    client = make_client()
    oai = openai.OpenAI()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, q in enumerate(mcqs, 1):
        print(f"  [{i}/{len(mcqs)}] {q['id']} ...", end=" ", flush=True)
        result = _run_question(q, oai=oai, client=client, embedder=embedder, log_path=LOG_PATH)
        if result["flags"]:
            status = f"FLAG({','.join(result['flags'])})"
        elif result["pass"]:
            status = "pass"
        else:
            status = "fail"
        print(status)
        results.append(result)
        output = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        OUTPUT_JSON.write_text(json.dumps(output, indent=2))

    print(f"\nResults written to {OUTPUT_JSON}")

    md = _build_md(results)
    OUTPUT_MD.write_text(md)
    print(f"Report written to {OUTPUT_MD}")

    p, d = _pass_rate(results)
    print(f"Pass rate: {_pct(p, d)}")


if __name__ == "__main__":
    main()
