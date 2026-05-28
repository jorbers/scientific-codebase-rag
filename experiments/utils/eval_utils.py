def is_hit(payload: dict, source_symbol: str) -> bool:
    class_name = payload.get("class_name", "")
    func_name = payload.get("function_name", "")
    full_name = f"{class_name}.{func_name}" if class_name else func_name
    method_only = source_symbol.split(".")[-1]
    return full_name == source_symbol


# ── retrieval config ───────────────────────────────────────────────────────────
TOP_K = 3   # results returned per query


def retrieve(*args, **kwargs):
    import inspect

    frame = inspect.currentframe()
    if frame is None:
        raise NameError("retrieve is not available")
    caller = frame.f_back
    while caller is not None:
        candidate = caller.f_globals.get("retrieve")
        if candidate is not None and candidate is not retrieve:
            return candidate(*args, **kwargs)
        caller = caller.f_back
    raise NameError("retrieve is not available")


# ── Eval helpers ───────────────────────────────────────────────────────────────


def run_eval(qa_pairs: list[dict], index: dict, k: int = TOP_K) -> tuple[dict, dict]:
    """Run recall@k over qa_pairs against index. Returns (hits, totals) by tier."""
    hits   = {"api": 0, "workflow": 0, "conceptual": 0}
    totals = {"api": 0, "workflow": 0, "conceptual": 0}
    for qa in qa_pairs:
        results     = retrieve(qa["query"], index, k=k, silent=True)
        totals[qa["tier"]] += 1
        if any(is_hit(r, qa["source_symbol"]) for r in results):
            hits[qa["tier"]] += 1
    return hits, totals


def print_recall(hits: dict, totals: dict, k: int = TOP_K, label: str = "") -> None:
    """Print recall breakdown by tier and overall."""
    header = f"── Recall@{k} by tier"
    if label:
        header += f" — {label}"
    print(header)
    for tier in ["api", "workflow", "conceptual"]:
        h, t = hits[tier], totals[tier]
        pct  = round(100 * h / t) if t else 0
        print(f"  {tier:<12} {h}/{t}  ({pct}%)")
    total_h = sum(hits.values())
    total_t = sum(totals.values())
    print(f"  {'overall':<12} {total_h}/{total_t}  ({round(100 * total_h / total_t) if total_t else 0}%)")


def eval_query(qa: dict, index: dict, k: int = TOP_K) -> None:
    """Print per-query retrieval results and hit/miss verdict."""
    results     = retrieve(qa["query"], index, k=k, silent=True)
    hit         = any(is_hit(r, qa["source_symbol"]) for r in results)

    print(f"\n{'='*70}")
    print(f"QUERY : {qa['query']}")
    print(f"TIER  : {qa['tier']}")
    print(f"TARGET: {qa['source_symbol']} ({qa['source_file']})")
    print(f"ANSWER: {qa['expected_answer']}")
    print(f"{'─'*70}")
    for i, r in enumerate(results, 1):
        print(f"  Rank {i}: {r['file_path']} — {r['function_name']}")
    print(f"{'─'*70}")
    print(f"VERDICT: {'✓ HIT' if hit else '✗ MISS'}")


def compare_versions(
    qa_pairs: list[dict],
    index_a: dict,
    index_b: dict,
    label_a: str = "a",
    label_b: str = "b",
    k: int = TOP_K,
) -> None:
    """Print queries where hit/miss status changed between two index versions."""
    for qa in qa_pairs:
        hit_a = any(is_hit(r, qa["source_symbol"])
                    for r in retrieve(qa["query"], index_a, k=k, silent=True))
        hit_b = any(is_hit(r, qa["source_symbol"])
                    for r in retrieve(qa["query"], index_b, k=k, silent=True))
        if hit_a != hit_b:
            status = "✓ -> ✗ REGRESSED" if hit_a else "✗ -> ✓ RECOVERED"
            print(f"{status}: {qa['source_symbol']}")


def diagnose_wide(qa_pairs: list[dict], index: dict, n: int = 50) -> None:
    """Dense-only wide retrieval diagnostic."""
    col = index["col"]
    for qa in qa_pairs:
        target_sym  = qa["source_symbol"].split(".")[-1]
        target_file = qa["source_file"].split("/")[-1]

        results_wide = col.query(query_texts=[qa["query"]], n_results=n)
        all_metas    = results_wide["metadatas"][0]

        target_rank = next(
            (i + 1 for i, m in enumerate(all_metas)
             if m["file_path"].split("/")[-1] == target_file
             and m["function_name"] == target_sym),
            None,
        )

        top10_docs = results_wide["documents"][0][:10]

        # "is the target at least reachable by context, even if not ranked directly?"
        referenced = any(target_sym in doc for doc in top10_docs)

        rank_str = str(target_rank) if target_rank else "not found"
        ref_str  = "✓" if referenced else "✗"
        print(f"  {qa['source_symbol']:<60}  dense: {rank_str:<10}  mentioned-in-top-10: {ref_str}")


def retrieve_wide(*args, **kwargs):
    """Proxy — resolved to the notebook-local retrieve_wide at call time."""
    import inspect
    frame = inspect.currentframe()
    if frame is None:
        raise NameError("retrieve_wide is not available")
    caller = frame.f_back
    while caller is not None:
        candidate = caller.f_globals.get("retrieve_wide")
        if candidate is not None and candidate is not retrieve_wide:
            return candidate(*args, **kwargs)
        caller = caller.f_back
    raise NameError("retrieve_wide is not available")


def diagnose_wide_hybrid(qa_pairs: list[dict], index: dict, n: int = 50) -> None:
    """Hybrid (BM25 + dense + RRF) wide retrieval diagnostic."""
    col = index["col"]
    for qa in qa_pairs:
        target_sym  = qa["source_symbol"].split(".")[-1]
        target_file = qa["source_file"].split("/")[-1]

        results = retrieve_wide(qa["query"], index, n=n)

        hybrid_rank = next(
            (i + 1 for i, m in enumerate(results)
             if m["file_path"].split("/")[-1] == target_file
             and m["function_name"] == target_sym),
            None,
        )

        top10_ids = [
            f"{m['file_path']}:{m['start_line']}-{m['end_line']}"
            for m in results[:10]
        ]
        fetched    = col.get(ids=top10_ids, include=["documents"])
        top10_docs = fetched["documents"]
        if len(top10_docs) < len(top10_ids):
            print(f"  warning: only {len(top10_docs)}/{len(top10_ids)} top-10 docs fetched")

        # "is the target at least reachable by context, even if not ranked directly?"
        referenced = any(target_sym in doc for doc in top10_docs)

        rank_str = str(hybrid_rank) if hybrid_rank else "not found"
        ref_str  = "✓" if referenced else "✗"
        print(f"  {qa['source_symbol']:<60}  hybrid: {rank_str:<10}  mentioned-in-top-10: {ref_str}")
