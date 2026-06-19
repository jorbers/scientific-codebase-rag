# Codebase RAG

A RAG layer for AI coding agents working with scientific codebases, exposed
via an MCP server. The core constraint: no hallucinated API calls. Every
generated function, class, and argument must exist in the actual codebase.

See [SETUP.md](SETUP.md) to get running.

---

## Architecture

**Two Qdrant collections, kept separate by design:**
- `kernelpack_code` — chunked Python source, multiple named vector spaces per point
- `kernelpack_papers` — paper chunks used as a vocabulary bridge between natural
  language queries and code identifiers

Cross-collection queries run as sequential legs in Python. Qdrant has no native
cross-collection chaining; the bridge is application-layer.

**Chunking:** tree-sitter at function/class boundaries (coarse chunks) with
statement-aligned 3–5 line windows within each function (fine chunks). Retrieval
targets fine chunks; generation receives the coarse parent. Each chunk also gets
a context header: parent class name, filtered imports, and neighbor signatures.

**Embeddings:** primary model is `jina-code-embeddings-0.5b`
(instruction-prefixed, NL2Code/Code2Code); `Qwen3-Embedding-0.6B` covers the
paper collection and NL-leaning query representations. UniXcoder is retained as
an ablation baseline. Each chunk is embedded into multiple named vector spaces
representing different text combinations (code only, code + summary, summary
only, etc.). Model swap does not require re-ingestion — payload stores all text
needed to rebuild vectors.

**Sparse vectors:** Qdrant native sparse with an identifier-aware tokenizer.
Emits intact dotted tokens (`PoissonSolver.solve`) and decomposed fragments
(`poisson`, `solver`, `solve`) so both identifier queries and natural-language
queries hit. IDF computed server-side.

**Point IDs:** deterministic uuid5 keyed on
`doc_type:source_file:qualname:granularity:window_idx`. Re-ingestion upserts
in place; no delete-before-insert.

---

## Data Flow

### Ingestion (source → Qdrant)

1. `chunk_file()` splits a `.py` file into coarse chunks at function/class
   boundaries via tree-sitter.
2. `fine_chunks()` produces 3–5 line statement-aligned windows within each
   coarse chunk.
3. `build_header()` attaches a context header to each coarse chunk (parent
   class, filtered imports, neighbor signatures).
4. `extract_metadata()` builds the symbol table, matches `math_terms` against
   the domain lexicon, and resolves `cross_refs` to other chunk UUIDs.
5. `summarize_chunk()` generates a plain-English LLM summary, cached to disk by
   content hash.
6. `build_representation()` assembles the text strings for each named vector
   space (e.g. `ctx` = summary + header + code, `code` = raw code only).
7. `JinaCodeEmbedder` / `Qwen3Embedder` produce dense vectors; `to_qdrant_sparse()`
   produces the BM25 sparse vector.
8. Points are batched and upserted into `kernelpack_code`.

Paper ingestion follows the same upsert pattern into `kernelpack_papers`, then a
second pass populates `math__qwen3` vectors on code points by linking each
chunk's `math_terms` back to the paper corpus.

### Retrieval (query → CodeChunk list)

The default plan (`hybrid`) runs entirely against `kernelpack_code`:

1. Embed the query with JinaCode (NL2Code instruction prefix).
2. Build a sparse BM25 vector from the query.
3. Qdrant prefetch: dense leg (`ctx__jinacode`) + sparse leg (`bm25_code`),
   fused via RRF → ranked `Candidate` list.
4. `fine_to_coarse()` swaps fine-chunk results for their coarse parent (larger
   context for generation).
5. `expand_cross_refs()` optionally appends cross-referenced chunks.

The `trimodal` plan adds a first leg against `kernelpack_papers` to extract
`math_terms`, then filters the code query by those terms before fusing.

---

## Module Map

| File | What it does | What calls it |
|---|---|---|
| `__main__.py` | CLI entry point; routes `python -m kernelpack_rag <subcommand>` | User / shell |
| `config.py` | `COLLECTIONS_CONFIG` — named vector spaces and dims for both collections | `schema.py`, `ingest.py`, `verify.py` |
| `schema.py` | `ensure_collections()` — idempotent collection creation, schema validation against `config.py`, int8 quantization, payload index creation | `ingest.py`, `verify.py` |
| `ingest.py` | Orchestrator: runs the full ingestion pipeline; also handles paper ingestion, `math__qwen3` population, `--prune`, and the invariant report | `__main__.py` |
| `retrieve.py` | Query plans returning `Candidate` lists: `hybrid`, `hybrid_filtered`, `trimodal`, `two_leg`, `expand_cross_refs`, `fine_to_coarse`. No CLI or global state. | `verify.py`, MCP server, notebooks |
| `verify.py` | Verification gate: collection invariants + golden-set recall@3/5/10 | `__main__.py` |
| `qlog.py` | Append-only JSONL event writer and per-`query_id` reader | MCP server, notebooks |
| `chunking/` | Coarse chunker, fine chunker, header builder, metadata extractor, paper loader | `ingest.py` |
| `embed/` | `Embedder` protocol and registry; representation builder; sparse vector builder; JinaCode, Qwen3, UniXcoder wrappers | `ingest.py`, `retrieve.py`, `verify.py` |
| `enrich/` | Per-chunk LLM summarizer with content-hash disk cache | `ingest.py` |

---

## Design Decisions

| ID | Decision |
|---|---|
| D1 | Embed `summary + context_header + raw_code` as primary vector. Store `llm_summary` and `context_header` separately in payload so re-embedding never requires re-running tree-sitter or the summarizer. |
| D2 | Qdrant native sparse vectors with identifier-aware tokenizer. Not SPLADE, not miniCOIL — both use web-text vocabulary and mangle code identifiers. |
| D3 | Papers in a separate collection. Sequential two-leg query. Co-location adds no benefit and contaminates sparse IDF statistics. |
| D4 | Named vector spaces declared write-once at collection creation. 15 spaces in `kernelpack_code`, 2 in `kernelpack_papers`. Empty spaces cost nothing; missing spaces require migration. |
| D5 | Deterministic point IDs via uuid5. No line numbers in the key (they shift on edits). |
| D6 | Append-only JSONL logs. Never edit records in place. |
| D7 | All fields needed to rebuild embedded text (`text`, `context_header`, `llm_summary`) stored explicitly in payload. |

---

## MCP Interface

```
retrieve_code(query: str, module_filter?: str, k?: int) -> list[CodeChunk]
```

`retrieve_code` is the only implemented tool. `explain_api`, `suggest_workflow`,
and `run_example` are registered stubs — Phase 3 scope.

See [SETUP.md](SETUP.md) for how to start the server and register it with an
MCP client.
