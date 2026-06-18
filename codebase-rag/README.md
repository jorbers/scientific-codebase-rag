# Codebase RAG — Project 1

A trustworthy RAG layer for AI coding agents working with scientific codebases, exposed via an MCP server. The design constraint: the agent must not hallucinate API calls. Every generated function, class, and argument must exist in the actual codebase.

See the top-level [README](../README.md) for project context and the full phase roadmap.

---

## Architecture

**Two Qdrant collections, kept separate by design:**
- `kernelpack_code` — chunked Python source, multiple named vector spaces per point
- `kernelpack_papers` — LaTeX paper chunks, used as a vocabulary bridge between natural language queries and code identifiers

Cross-collection queries run as sequential legs in Python. Qdrant has no native cross-collection chaining; the bridge is application-layer by design.

**Retrieval is trimodal:** a query may need to match against code structure, mathematical terminology, and cross-references simultaneously. The pipeline handles this through named vector spaces (one embedding per representation per chunk), payload filtering on `math_terms`, and a two-leg retrieval pattern that uses the paper corpus to surface math terms before querying the code collection.

**Chunking:** tree-sitter at function/class boundaries with context headers (parent class, filtered imports, neighbor signatures). Fine chunks (3–5 line windows) and coarse chunks (full function) are both indexed; retrieval targets fine, generation receives coarse.

**Embeddings:** primary model is `jina-code-embeddings-0.5b` (instruction-prefixed, NL2Code/Code2Code); `Qwen3-Embedding-0.6B` covers the paper collection and NL-leaning queries. UniXcoder is retained as an ablation baseline. Model swap does not require re-ingestion — payload stores everything needed to rebuild embedded text.

**Sparse vectors:** Qdrant native sparse with an identifier-aware tokenizer. Emits intact dotted tokens (`PoissonSolver.solve`) and decomposed fragments (`poisson`, `solver`, `solve`) so both identifier queries and natural-language queries hit. IDF computed server-side.

**Point IDs:** deterministic uuid5 keyed on `doc_type:source_file:qualname:granularity:window_idx`. Re-ingestion upserts in place; no delete-before-insert.

**Logging:** append-only JSONL event stream correlated by `query_id`. Events: `retrieval`, `generation`, `execution`. Distinguishes identifiers absent from retrieved chunks (retrieval miss) from identifiers absent from the package entirely (true hallucination).

---

## Architecture Decisions

| ID | Decision |
|---|---|
| D1 | Embed `summary + context_header + raw_code` as primary vector. Store `llm_summary` and `context_header` separately in payload so re-embedding never requires re-running tree-sitter or the summarizer. |
| D2 | Qdrant native sparse vectors with identifier-aware tokenizer. Not SPLADE, not miniCOIL — both use web-text vocabulary and mangle code identifiers. |
| D3 | Papers in a separate collection (`kernelpack_papers`). Sequential two-leg query. Co-location adds no benefit and contaminates sparse IDF statistics. |
| D4 | Named vector spaces declared write-once at collection creation. 15 spaces in `kernelpack_code`, 2 in `kernelpack_papers`. Empty spaces cost nothing; missing spaces require migration. |
| D5 | Deterministic point IDs via uuid5. No line numbers (shift on edits). |
| D6 | Append-only JSONL logs. Never edit records in place. |
| D7 | All payload fields required to rebuild embedded text (`text`, `context_header`, `llm_summary`) stored explicitly in payload. |

---

## Pipeline Architecture

| File | What it does | What calls it |
|---|---|---|
| `__main__.py` | CLI entry point; routes `python -m kernelpack_rag <subcommand>` to the matching module. | User / shell |
| `config.py` | Holds `COLLECTIONS_CONFIG`, the write-once dict of named dense/sparse spaces and dims for both collections. | `schema.py`, `ingest.py`, `verify.py` |
| `schema.py` | `ensure_collections()` — idempotent collection creation, schema validation against `config.py` (hard-fails on drift), int8 quantization on 1024-dim spaces, and payload-index creation. | `ingest.py`, `verify.py` |
| `ingest.py` | Orchestrator: chunk → header → fine → metadata → summarize → represent → embed → batched upsert; also paper ingestion, `math__qwen3` population, `--prune`, and the invariant report. | `__main__.py` (`ingest`); helpers imported by `retrieve.py`, `verify.py` |
| `retrieve.py` | Query plans returning `Candidate` lists: `hybrid`, `hybrid_filtered`, `trimodal`, `two_leg`, `expand_cross_refs`, `fine_to_coarse`. No CLI or global state. | `verify.py`, MCP server, notebooks |
| `verify.py` | Verification gate: collection invariants (schema, missing primary vectors, resolvable `parent_id`/`math_source_id`) plus golden-set recall@3/5/10 against the parity baseline. | `__main__.py` (`verify`) |
| `qlog.py` | Append-only JSONL event writer (`write_event`) and per-`query_id` `fold` reader for retrieval/generation/execution events. | Generation layer, notebooks |
| `chunking/` | Coarse tree-sitter chunker, statement-aligned fine chunker, context header builder, metadata extractor (symbol table, `math_terms`, `cross_refs`), and paper `.md` loader. | `ingest.py` |
| `embed/` | `Embedder` protocol and registry; `ctx`/`code`/`codecom`/`com` representation builder; identifier-aware sparse vector builder; JinaCode, Qwen3, and UniXcoder model wrappers. | `ingest.py`, `retrieve.py`, `verify.py` |
| `enrich/` | Per-chunk LLM summary with content-hash disk cache. | `ingest.py` |

---

## MCP Interface

The retrieval pipeline is exposed as MCP tools so coding agents can query it before generating library code:

```
retrieve_code(query: str, module_filter?: str) -> list[CodeChunk]
explain_api(symbol_name: str) -> str
suggest_workflow(goal: str) -> WorkflowPlan
run_example(workflow_plan: WorkflowPlan) -> CodeSnippet
```

All logic lives in the retrieval pipeline. The MCP layer handles protocol and routing only.