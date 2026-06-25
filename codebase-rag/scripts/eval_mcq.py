#!/usr/bin/env python3
"""MCQ retrieval evaluation for KernelPack RAG pipeline.

Runs all 100 MCQ questions through the hybrid retrieval plan (k=10) and
checks whether any top-k result covers the cited source_refs.

Usage:
    python eval_mcq.py [--json out.json]

Metrics:
    recall@5, recall@10 — overall and by difficulty
    Adjusted metrics excluding structurally-impossible questions (all
    source_refs unindexed in Qdrant).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient

from kernelpack_rag.config import CODE_COLLECTION, make_client
from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
from kernelpack_rag.retrieve import hybrid, Candidate

# Derive from KP_SRC env var: KP_SRC=.../kernelpack-python/src/kernelpack → prefix=.../kernelpack-python/
_kp_src = os.environ.get("KP_SRC", "")
KERNELPACK_PREFIX = str(Path(_kp_src).parent.parent) + "/" if _kp_src else ""

# Code files confirmed indexed in Qdrant (relative to KERNELPACK_PREFIX).
# Derived from a live scroll of the collection on 2026-06-23.
# Includes __init__.py (module_exports chunks) and tests/ added 2026-06-23.
INDEXED_CODE_FILES: frozenset[str] = frozenset(
    {
        "src/kernelpack/__init__.py",
        "src/kernelpack/_numba.py",
        "src/kernelpack/divfree/__init__.py",
        "src/kernelpack/divfree/core.py",
        "src/kernelpack/domain/__init__.py",
        "src/kernelpack/domain/core.py",
        "src/kernelpack/domain/dual.py",
        "src/kernelpack/geometry/__init__.py",
        "src/kernelpack/geometry/core.py",
        "src/kernelpack/nodes/__init__.py",
        "src/kernelpack/nodes/core.py",
        "src/kernelpack/nodes/dual.py",
        "src/kernelpack/poly/__init__.py",
        "src/kernelpack/poly/core.py",
        "src/kernelpack/rbffd/__init__.py",
        "src/kernelpack/rbffd/core.py",
        "src/kernelpack/solvers/__init__.py",
        "src/kernelpack/solvers/_common.py",
        "src/kernelpack/solvers/_pu.py",
        "src/kernelpack/solvers/detail/__init__.py",
        "src/kernelpack/solvers/detail/incompressible_euler_bdf_backend.py",
        "src/kernelpack/solvers/diffusion.py",
        "src/kernelpack/solvers/heterogeneous_multispecies_diffusion.py",
        "src/kernelpack/solvers/incompressible_euler.py",
        "src/kernelpack/solvers/multispecies_diffusion.py",
        "src/kernelpack/solvers/nonlinear_variable_poisson.py",
        "src/kernelpack/solvers/poisson.py",
        "src/kernelpack/solvers/pu_diffusion.py",
        "src/kernelpack/solvers/pu_multispecies.py",
        "src/kernelpack/solvers/pu_sl_advection.py",
        "src/kernelpack/solvers/pu_sl_fd_advection_diffusion.py",
        "src/kernelpack/solvers/pu_sl_multispecies.py",
        "src/kernelpack/solvers/pu_sl_pu_advection_diffusion.py",
        "src/kernelpack/solvers/variable_poisson.py",
        "tests/test_convergence_examples.py",
        "tests/test_divfree.py",
        "tests/test_incompressible_euler.py",
        "tests/test_nodes_rbffd.py",
        "tests/test_poly.py",
        "tests/test_pu_solvers.py",
        "tests/test_solvers.py",
    }
)


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _parse_source_ref(ref: str) -> tuple[str, int | None, int | None]:
    """'src/foo/bar.py:10-25' → ('src/foo/bar.py', 10, 25)"""
    parts = ref.split(":")
    file_path = parts[0]
    if len(parts) < 2:
        return file_path, None, None
    line_part = parts[1]
    if "-" in line_part:
        lo, hi = line_part.split("-", 1)
        return file_path, int(lo), int(hi)
    line = int(line_part)
    return file_path, line, line


def _is_indexed(ref_file: str) -> bool:
    if ref_file == "README.md":
        return True
    return ref_file in INDEXED_CODE_FILES


def _chunk_matches_ref(payload: dict, ref: str) -> bool:
    """True if this chunk payload covers the region cited in ref."""
    ref_file, ref_start, ref_end = _parse_source_ref(ref)
    source_file: str = payload.get("source_file", "") or ""

    # README: match by filename only; line_range is always [0,0]
    if ref_file == "README.md":
        return source_file == "README.md"

    # Code: source_file is an absolute path; ref_file is relative
    if not source_file.endswith(ref_file):
        return False

    # Line-range overlap check
    if ref_start is not None and ref_end is not None:
        chunk_lr = payload.get("line_range") or [0, 0]
        c_start, c_end = int(chunk_lr[0]), int(chunk_lr[1])
        if c_start == 0 and c_end == 0:
            # Unknown line range — accept file-level match
            return True
        # Reject non-overlapping ranges
        if c_end < ref_start or c_start > ref_end:
            return False

    return True


def _any_hit(candidates: list[Candidate], refs: list[str], k: int) -> bool:
    return any(
        _chunk_matches_ref(c.payload, ref)
        for c in candidates[:k]
        for ref in refs
    )


# ---------------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------------


def _evaluate(q: dict, candidates: list[Candidate]) -> dict:
    refs = q["source_refs"]
    ref_files = [_parse_source_ref(r)[0] for r in refs]
    indexed_flags = [_is_indexed(f) for f in ref_files]

    all_unindexed = not any(indexed_flags)
    all_indexed = all(indexed_flags)

    hit5 = _any_hit(candidates, refs, 5)
    hit10 = _any_hit(candidates, refs, 10)

    return {
        "id": q["id"],
        "difficulty": q["difficulty"],
        "category": q["category"],
        "question": q["question"],
        "source_refs": refs,
        "indexability": (
            "none" if all_unindexed
            else "full" if all_indexed
            else "partial"
        ),
        "hit_at_5": hit5,
        "hit_at_10": hit10,
        "top10_sources": [
            _short_source(c.payload) for c in candidates[:10]
        ],
        "top10_previews": [
            _text_preview(c.payload) for c in candidates[:10]
        ],
    }


def _short_source(payload: dict) -> str:
    sf = payload.get("source_file", "") or ""
    if sf.startswith(KERNELPACK_PREFIX):
        sf = sf[len(KERNELPACK_PREFIX):]
    fn = payload.get("function_name") or ""
    return f"{sf}:{fn}" if fn else sf


def _text_preview(payload: dict, max_chars: int = 250) -> str:
    text = (payload.get("text") or "").strip()
    # Strip markdown hyperlinks — keep the link text, drop the URL
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _clean_stored_preview(raw: str, max_chars: int = 250) -> str:
    """Strip markdown links and truncate — applied to previews at display time."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw).strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _source_with_preview(source: str, previews: list[str]) -> str:
    """For the per-question table: if source is README, append a short snippet."""
    if not source.endswith("README.md") or not previews:
        return source
    snippet = _clean_stored_preview(previews[0], max_chars=80)
    return f"{source} — {snippet}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "0/0"


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f" {title}")
    print(f"{'─'*60}")


def _stats(results: list[dict]) -> dict:
    answerable = [r for r in results if r["indexability"] != "none"]
    unanswerable = [r for r in results if r["indexability"] == "none"]
    return {
        "total": len(results),
        "full": sum(1 for r in results if r["indexability"] == "full"),
        "partial": sum(1 for r in results if r["indexability"] == "partial"),
        "none": len(unanswerable),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "h5_all": sum(r["hit_at_5"] for r in results),
        "h10_all": sum(r["hit_at_10"] for r in results),
        "h5_ans": sum(r["hit_at_5"] for r in answerable),
        "h10_ans": sum(r["hit_at_10"] for r in answerable),
    }


def print_report(results: list[dict]) -> None:
    s = _stats(results)
    total = s["total"]
    answerable = s["answerable"]
    unanswerable = s["unanswerable"]

    print("\n" + "="*60)
    print(" KernelPack MCQ Retrieval Eval — hybrid plan, k=10")
    print("="*60)

    _section("Pre-flight: indexability")
    print(f"  Total questions:                {total}")
    print(f"  All source_refs indexed:        {s['full']}")
    print(f"  Some source_refs indexed:       {s['partial']}")
    print(f"  ALL source_refs UNINDEXED:      {s['none']}")
    if unanswerable:
        print(f"\n  ⚠  {s['none']} questions cite only unindexed files — pipeline ceiling issues.")

    _section(f"Overall recall (all {total} questions)")
    print(f"  recall@5:   {_pct(s['h5_all'], total)}")
    print(f"  recall@10:  {_pct(s['h10_all'], total)}")

    if s["none"] > 0:
        _section(f"Recall excluding ceiling misses ({len(answerable)} questions)")
        print(f"  recall@5:   {_pct(s['h5_ans'], len(answerable))}")
        print(f"  recall@10:  {_pct(s['h10_ans'], len(answerable))}")

    _section("By difficulty")
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        ans_sub = [r for r in answerable if r["difficulty"] == diff]
        h5 = sum(r["hit_at_5"] for r in subset)
        h5a = sum(r["hit_at_5"] for r in ans_sub)
        print(
            f"  {diff:<8}  all={_pct(h5, len(subset)):<22}"
            f"  answerable={_pct(h5a, len(ans_sub))}"
        )

    _section("By source type (all questions)")
    readme_qs = [r for r in results if any("README" in rf for rf in r["source_refs"])]
    code_only_qs = [r for r in results if not any("README" in rf for rf in r["source_refs"])]
    print(f"  README-citing questions:  {len(readme_qs)}  recall@5={_pct(sum(r['hit_at_5'] for r in readme_qs), len(readme_qs))}")
    print(f"  Code-only questions:      {len(code_only_qs)}  recall@5={_pct(sum(r['hit_at_5'] for r in code_only_qs), len(code_only_qs))}")

    _section("Misses among answerable questions (recall@5)")
    misses = [r for r in answerable if not r["hit_at_5"]]
    if not misses:
        print("  None — 100% recall@5 on answerable questions!")
    else:
        for r in misses:
            print(f"  [{r['id']}] [{r['difficulty']:<6}] {r['question'][:70]}")
            print(f"           refs: {r['source_refs']}")
            print(f"           top5: {r['top10_sources'][:5]}")

    if unanswerable:
        _section("Unindexed questions (always miss)")
        for r in unanswerable:
            print(f"  [{r['id']}] [{r['difficulty']:<6}] refs={r['source_refs']}")

    print()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _build_md(results: list[dict]) -> str:
    import datetime
    s = _stats(results)
    total = s["total"]
    answerable = s["answerable"]
    unanswerable = s["unanswerable"]
    misses5 = [r for r in answerable if not r["hit_at_5"]]
    misses10 = [r for r in answerable if not r["hit_at_10"]]

    lines: list[str] = []
    ts = datetime.datetime.now().strftime("%Y-%m-%d")

    headline_rows = [
        f"| recall@5  ({total} questions) | {_pct(s['h5_all'], total)} |",
        f"| recall@10 ({total} questions) | {_pct(s['h10_all'], total)} |",
        f"| Misses at k=5  | {len(misses5)} |",
        f"| Misses at k=10 | {len(misses10)} |",
    ]
    if s["none"] > 0:
        headline_rows += [
            f"| Ceiling misses (unindexed refs) | {s['none']} |",
            f"| recall@5  (indexed questions only) | {_pct(s['h5_ans'], len(answerable))} |",
            f"| recall@10 (indexed questions only) | {_pct(s['h10_ans'], len(answerable))} |",
        ]

    lines += [
        "# KernelPack RAG — Retrieval Quality Report",
        f"*{ts} · hybrid retrieval · k=10 · {total} questions*",
        "",
        "---",
        "",
        "## Headline metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
        *headline_rows,
        "",
        "> **recall@k** — fraction of questions where the correct source chunk",
        "> appeared in the top-k retrieval results, with no LLM assist.",
        "> A miss means the model has no retrieved grounding for that question.",
        "",
        "---",
        "",
        "## By difficulty",
        "",
        "| Difficulty | n | recall@5 | recall@10 | misses@5 |",
        "|---|---|---|---|---|",
    ]
    for diff in ("easy", "medium", "hard"):
        sub = [r for r in results if r["difficulty"] == diff]
        ans = [r for r in answerable if r["difficulty"] == diff]
        h5 = sum(r["hit_at_5"] for r in sub)
        h10 = sum(r["hit_at_10"] for r in sub)
        m5 = sum(1 for r in ans if not r["hit_at_5"])
        lines.append(f"| {diff} | {len(sub)} | {_pct(h5, len(sub))} | {_pct(h10, len(sub))} | {m5} |")
    lines += [""]

    lines += [
        "---",
        "",
        "## By category",
        "",
        "| Category | n | recall@5 | recall@10 |",
        "|---|---|---|---|",
    ]
    cats: dict[str, list] = defaultdict(list)
    for r in results:
        cats[r["category"]].append(r)
    for cat, qs in sorted(cats.items()):
        h5 = sum(r["hit_at_5"] for r in qs)
        h10 = sum(r["hit_at_10"] for r in qs)
        lines.append(f"| {cat} | {len(qs)} | {_pct(h5, len(qs))} | {_pct(h10, len(qs))} |")
    lines += [""]

    idx_summary = (
        f"All {total} questions have at least one source ref in the indexed collection."
        if s["none"] == 0
        else f"{len(answerable)} of {total} questions have at least one source ref indexed."
        f" {s['none']} have no refs in the collection (ceiling misses — retrieval can never win)."
    )
    lines += [
        "---",
        "",
        "## Indexability",
        "",
        idx_summary,
        "",
    ]
    if s["none"] > 0:
        lines += [
            "| Status | Count |",
            "|---|---|",
            f"| All refs indexed | {s['full']} |",
            f"| Some refs indexed | {s['partial']} |",
            f"| No refs indexed (ceiling miss) | {s['none']} |",
            "",
            "### Ceiling misses (refs not in collection)",
            "",
            "| ID | Difficulty | Source refs |",
            "|---|---|---|",
        ]
        for r in unanswerable:
            refs = ", ".join(r["source_refs"])
            lines.append(f"| {r['id']} | {r['difficulty']} | `{refs}` |")
        lines += [""]

    lines += [
        "---",
        "",
        "## Retrieval misses at k=5",
        "",
    ]
    if not misses5:
        lines += ["*None — 100% recall@5.*", ""]
    else:
        lines += [
            f"*{len(misses5)} questions where the correct chunk was not in the top 5 results.*",
            "",
        ]
        for r in misses5:
            top_src = r["top10_sources"][0] if r["top10_sources"] else "—"
            raw_preview = r.get("top10_previews", [None])[0] or ""
            preview = _clean_stored_preview(raw_preview) if raw_preview else ""
            refs = "; ".join(r["source_refs"])
            lines += [
                f"**{r['id']}** · {r['difficulty']} · `{r['category']}`",
                "",
                f"Q: {r['question']}",
                f"Expected: `{refs}`",
                f"Retrieved: `{top_src}`",
            ]
            if preview:
                lines.append(f"> {preview}")
            lines.append("")

    lines += [
        "---",
        "",
        "## Per-question results",
        "",
        "| ID | Difficulty | Category | hit@5 | hit@10 | Top retrieved |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        hit5 = "✓" if r["hit_at_5"] else "✗"
        hit10 = "✓" if r["hit_at_10"] else "✗"
        top_src = r["top10_sources"][0] if r["top10_sources"] else "—"
        top_previews = r.get("top10_previews", [])
        top = _source_with_preview(top_src, top_previews).replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['difficulty']} | {r['category']} "
            f"| {hit5} | {hit10} | {top} |"
        )
    lines += [""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCQ retrieval eval for KernelPack RAG.")
    parser.add_argument(
        "--json",
        type=Path,
        metavar="OUT",
        help="Write full per-question results to this JSON file.",
    )
    parser.add_argument(
        "--md",
        type=Path,
        metavar="OUT",
        help="Write a markdown retrieval quality report to this file.",
    )
    parser.add_argument(
        "--mcq",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to MCQ bank JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mcqs: list[dict] = json.loads(args.mcq.read_text())

    client = make_client()
    embedder = JinaCodeEmbedder()

    print(f"Loaded {len(mcqs)} questions. Running hybrid retrieval (k=10)...")
    t0 = time.monotonic()

    results: list[dict] = []
    for i, q in enumerate(mcqs, 1):
        candidates = hybrid(
            q["question"],
            client=client,
            collection=CODE_COLLECTION,
            embedder=embedder,
            k=10,
        )
        results.append(_evaluate(q, candidates))
        if i % 20 == 0 or i == len(mcqs):
            elapsed = time.monotonic() - t0
            print(f"  {i}/{len(mcqs)} done  ({elapsed:.1f}s)")

    elapsed = time.monotonic() - t0
    print(f"\nRetrieval complete in {elapsed:.1f}s ({elapsed/len(mcqs)*1000:.0f}ms/query)")

    print_report(results)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"Per-question results written to {args.json}")

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(_build_md(results))
        print(f"Retrieval quality report written to {args.md}")


if __name__ == "__main__":
    main()
