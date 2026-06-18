# Codebase RAG — Project 1

A trustworthy RAG layer for AI coding agents working with scientific codebases, exposed via an MCP server. The design constraint: the agent must not hallucinate API calls. Every generated function, class, and argument must exist in the actual codebase.

See the top-level [README](../README.md) for project context and the full phase roadmap.

---

## What This Builds

A retrieval pipeline over `kernelpack-python` that a coding agent (Codex) queries before generating any library code. The agent operates only on what the RAG surfaces — it cannot answer from training data alone. Generation failures are attributed back to specific retrieval gaps via a sandboxed execution harness.

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

The pipeline has two entry points, both reached through `python -m kernelpack_rag <subcommand>`. **Ingestion** (`ingest.py`) walks the `kernelpack` source tree once per run: it parses each `.py` file into coarse chunks (functions, methods, class headers, module docstrings) and statement-aligned fine chunks via tree-sitter, builds a context header per chunk, generates a cached LLM summary, extracts metadata (`math_terms`, `cross_refs`, `has_numba`) against a package-wide symbol table, turns each chunk into one or more text representations, embeds those into the requested named vector spaces (plus an identifier-aware sparse BM25 vector), and upserts the points into Qdrant under deterministic uuid5 IDs so re-runs overwrite in place. Optionally it ingests a paper corpus into `kernelpack_papers` and copies each linked paper vector into a code point's `math__qwen3` space. **Retrieval** (`retrieve.py`) exposes plain callables — `hybrid`, `hybrid_filtered`, `trimodal`, `two_leg`, `expand_cross_refs`, `fine_to_coarse` — that issue Qdrant Query-API requests (dense + sparse prefetch fused with RRF), optionally rerank the candidates, and return a uniform `Candidate` list. `verify.py` gates the index by checking collection invariants and golden-set recall; `qlog.py` is the append-only event log the generation layer is intended to write through.

One row per source file in `kernelpack_rag/`:

| File | What it does | What calls it |
|---|---|---|
| `__main__.py` | CLI entry point; routes `python -m kernelpack_rag <subcommand>` to the matching module. | User / shell |
| `config.py` | Holds `COLLECTIONS_CONFIG`, the write-once dict of named dense/sparse spaces and dims for both collections. | `schema.py`, `ingest.py`, `verify.py` |
| `schema.py` | `ensure_collections()` — idempotent collection creation, schema validation against `config.py` (hard-fails on drift), int8 quantization on 1024-dim spaces, and payload-index creation. | `ingest.py`, `verify.py` |
| `tokenizer.py` | Pure identifier-aware tokenization: emits intact dotted tokens plus snake/camel fragments, and hashes them to u32 sparse indices. | `embed/sparse.py`, `chunking/metadata.py` |
| `ingest.py` | Orchestrator: chunk → header → fine → metadata → summarize → represent → embed → batched upsert; also paper ingestion, `math__qwen3` population, `--prune`, and the invariant report. Hosts shared Qdrant helpers and collection-name constants. | `__main__.py` (`ingest`); helpers imported by `retrieve.py`, `verify.py` |
| `retrieve.py` | Query plans returning `Candidate` lists: `hybrid`, `hybrid_filtered`, `trimodal`, `two_leg`, `expand_cross_refs`, `fine_to_coarse`. No CLI or global state — meant to be wrapped by the MCP server. | `verify.py`; `rerank.py` (`Candidate`); MCP server / notebooks |
| `rerank.py` | Reranker protocol: `NoopReranker` (default) and `CrossEncoderReranker` (cross-encoder, lexical fallback). | `retrieve.py`, `verify.py` |
| `qlog.py` | Append-only JSONL event writer (`write_event`) and per-`query_id` `fold` reader for retrieval/generation/execution events. | Generation/logging layer, notebooks |
| `verify.py` | Verification gate: collection invariants (schema, missing primaries, resolvable `parent_id`/`math_source_id`) plus golden-set recall@3/5/10 against the parity baseline. | `__main__.py` (`verify`) |
| `chunking/coarse.py` | Tree-sitter coarse chunker → `CoarseChunk` (function/method/class-header/module-docstring), with `MIN_LINES=5` merge behavior. | `ingest.py`, `chunking/fine.py`, `chunking/header.py` |
| `chunking/fine.py` | Splits a coarse function/method into 3–5 line statement-aligned `FineChunk` windows, each carrying its coarse `parent_id`. | `ingest.py` |
| `chunking/header.py` | `build_header()` — context header of parent class, referenced imports, and neighbor signatures, rendered as `#` comment lines. | `ingest.py` |
| `chunking/metadata.py` | Two-pass metadata: builds and serializes the qualname→uuid symbol table, then extracts `math_terms`/`cross_refs`/`has_numba` per chunk; owns the deterministic uuid5 helpers. | `ingest.py`, `chunking/fine.py` |
| `chunking/papers.py` | `load_paper_chunks()` — loads `.md` paper chunks plus sidecar `.json` metadata (`math_terms`, `equation_labels`, `section`) into `PaperChunk`s. | `ingest.py` |
| `embed/base.py` | `Embedder` protocol (`name`, `dim`, `embed_batch`, `embed_query_batch`) and `EmbedderRegistry`. | `ingest.py`, embedder implementations |
| `embed/representations.py` | `build_representation()` — builds `ctx`/`code`/`codecom`/`com` text variants from payload; returns `None` to skip a space (e.g. empty `com`). | `ingest.py` |
| `embed/sparse.py` | `build_sparse_vector` / `to_qdrant_sparse` — turn text into a Qdrant `SparseVector` of raw term frequencies (IDF applied server-side). | `ingest.py`, `retrieve.py` |
| `embed/jinacode.py` | `JinaCodeEmbedder` — primary code model (jina-code-0.5b, dim 896), instruction-prefixed query/passage encoding. | `ingest.py`, `verify.py` |
| `embed/qwen.py` | `QwenEmbedder` — Qwen3-Embedding-0.6B (dim 1024) for papers, summary, math, and NL-leaning queries. | `ingest.py` |
| `embed/unixcoder.py` | `UniXcoderEmbedder` — UniXcoder (dim 768), retained for the ablation matrix. | `ingest.py` |
| `enrich/summarize.py` | `summarize_chunk()` — per-coarse-chunk LLM summary cached on disk by content hash. | `ingest.py` |

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