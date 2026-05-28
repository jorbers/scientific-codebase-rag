# Scientific Codebase RAG — Research Log

> Retrieval-Augmented Generation for scientific computing libraries.
> Active research — updated as experiments unfold.

This repository documents ongoing research into RAG systems purpose-built for scientific codebases. The work targets a failure mode common to existing RAG systems: they treat scientific code like generic software. But scientific code carries a **trimodal retrieval problem** — *code*, *mathematics*, and *cross-references* must be resolved simultaneously. Standard dense retrieval fails here because the user's vocabulary ("4th-order Laplacian") and the codebase's naming conventions (`assemble_op`, `FDDiffOp`) are disjoint — and no amount of embedding tuning bridges that gap alone.

The primary testbed is **[kernelpack-python](https://github.com/ShankarLab/kernelpack-python)** — a Python library for RBF-FD (radial basis function finite difference) numerical methods. It is representative of the class of scientific libraries this research targets: domain-specific mathematical vocabulary, a naming convention orthogonal to natural language, and a user community that knows what they want numerically but not the library's API.

---

## Two Projects, One Foundation

Two RAG systems are under development in parallel. They share retrieval infrastructure through Phase 3, then diverge at the generation and interface layers.

### Project 1 — Developer / Coding Agent RAG

A trustworthy RAG layer for AI coding agents (e.g., Codex, Cursor) working with scientific codebases. The core design constraint: the agent must not hallucinate API calls. Every generated function, class, and argument must exist in the actual codebase. This is enforced through constrained retrieval — the agent operates only on what the RAG surfaces — and evaluated via a sandboxed execution harness that attributes generation failures back to specific retrieval gaps (Phase 5).

**Primary user:** developers and researchers using AI coding assistants with scientific libraries.

### Project 2 — Citizen Science RAG

Scientific simulation has always had an unspoken prerequisite: you had to already
speak the language. Knowing that "heat spreading through a material" is a diffusion
equation, that an "irregular boundary" is a non-convex domain, that the right tool
is an RBF-FD solver — none of that is obvious, and all of it has historically been
the price of entry.

This project removes that prerequisite. This is what a textbook of the future looks
like — not static explanations of theory, but a system that meets you where you are
and hands you working code for whatever you're curious about. A user who asks
*"how do I simulate what happens when a nuclear reactor overheats?"* should get a
working, grounded answer — not a wall of jargon, not hallucinated pseudocode, and
not a closed door.

The retrieval infrastructure is shared with Project 1. What differs is the generation
layer — more scaffolded, more pedagogical — and the interface, which is designed for
someone who has never heard of a solver or a domain descriptor.

**Primary user:** anyone curious enough to ask.

The two projects share all early-phase retrieval work (chunking, indexing, embedding evaluation, hybrid search). They diverge at Phase 3 (generation prompting strategy) and Phase 4 (MCP/interface layer).

---

## Repo Structure

```
scientific-codebase-rag/
│
├── README.md                        ← this file
│
├── experiments/                     ← shared retrieval experimentation phase
│   ├── README.md
│   ├── 01_naive_to_hybrid.ipynb
│   ├── 02a_eval_bm25_mangling.ipynb
│   ├── 02b_eval_baseline.ipynb
│   ├── 03_qdrant_migration.ipynb
│   ├── qa_pairs/
│   │   └── solvers_qa.json
│   ├── metadata/
│   │   └── solvers_summaries.json
│   ├── scripts/
│   │   └── migrate_to_qdrant.py
│   ├── utils/
│   │   └── chunking.py
│   │   └── eval_utils.py
│   └── reports/
│       ├── 02_eval_report_v2.pdf
│       └── 03_qdrant_migration_report.pdf
│
├── codebase-rag/                    ← Project 1 (fills in as pipeline matures)
│   └── README.md
│
└── citizen-science-rag/             ← Project 2 (fills in as pipeline matures)
    └── README.md
```

---

## Notebooks

| # | Notebook | Investigates | Key Finding |
|---|---|---|---|
| 01 | [`01_naive_to_hybrid`](experiments/01_naive_to_hybrid.ipynb) | Chunking strategy, embedding model selection, hybrid retrieval construction | Fixed-line chunking and general-purpose embeddings both fail on scientific code; AST-aware chunking + UniXcoder + BM25/RRF hybrid establishes the right foundation |
| 02a | [`02a_eval_bm25_mangling`](experiments/02a_eval_bm25_mangling.ipynb) | Research artifact — preserves broken whitespace BM25 tokenization to isolate the sparse leg's contribution by comparison with 02b | Whitespace tokenization renders BM25 functionally dead; recall@3 at v3 = 4/10 vs. 8/10 with correct tokenization — a 40-point gap from one incorrect function call |
| 02b | [`02b_eval_baseline`](experiments/02b_eval_baseline.ipynb) | Authoritative retrieval experiments v1→v3 with identifier-aware BM25 tokenization — eval harness, failure mode diagnosis, LLM summary enrichment | Vocabulary gap is closable via LLM summary enrichment (recall@5 = 10/10 after v3); bottleneck is ranking, not coverage; cross-encoder reranker is the highest-leverage next step |
| 03 | [`03_qdrant_migration`](experiments/03_qdrant_migration.ipynb) | ChromaDB → Qdrant migration, point-by-point verification, hybrid parity confirmation, cross-encoder reranker preview | Migration verified clean; hybrid parity confirmed (8/10, 10/10, 10/10 — matches 02b exactly); reranker adds +5 recall@3 over dense-only Qdrant |

---

## Where Things Stand

*This table is versioned — updated as the research unfolds. Last updated: 05-28-2026.*

| Component | Current State | Problem |
|---|---|---|
| Chunking | Tree-sitter at function/class boundaries, MIN_LINES=5 | Good, but no hierarchy — no "read big" after "retrieve small" |
| Vector store | **Qdrant** (migration complete, parity verified) | Manual BM25+RRF not yet replaced with Qdrant native sparse hybrid |
| Embeddings | UniXcoder (microsoft/unixcoder-base) + LLM-generated summaries prepended | Summaries closed the vocabulary gap; embedding model itself not yet ablated |
| Search | BM25 + dense + RRF fusion (Qdrant, mirrored from 02b) | Works; migrate to Qdrant native sparse vectors before treating as production |
| Reranking | **Implemented** — `cross-encoder/ms-marco-MiniLM-L-6-v2` | Ranking quality for generation untested; hybrid+reranker not yet evaluated together |
| LLM generation | None | Retrieval-only; not yet end-to-end |
| Eval | 10 Q&A pairs (solvers module only, Codex-generated) | Narrow scope; not yet domain-expert validated — numbers are directional |
| Metadata | Minimal (file path, function name, class name) | Missing math context, cross-references, chunk type |

**Key findings from current experiments:**
- Vocabulary gap is closable: after LLM summary enrichment, all 10 targets land in the top-50 dense neighborhood. Recall@5 = 10/10 in v3.
- Bottleneck is ranking, not retrieval coverage. A cross-encoder reranker adds +5 recall@3 over dense-only Qdrant (3/10 → 8/10).
- BM25 tokenization matters: whitespace-only tokenization kills the sparse leg silently. Identifier-aware tokenization is non-optional.

---

## The Plan (Phase by Phase)

---

### Phase 0 — Eval Framework

**Status: partially complete.**

Rigorous eval is the foundation for everything else — a change is only an improvement if there is a baseline to compare against.

#### 0.1 — Q&A Eval Set

**Done (partial):** `qa_pairs/solvers_qa.json` — 10 pairs covering `kernelpack.solvers`, generated by Codex, three tiers (api, workflow, conceptual).

**Remaining:**
- Expand to all 6 modules (`geometry`, `nodes`, `domain`, `poly`, `rbffd`, `solvers`) — target 50–100 pairs total
- Make pairs longer and more specific — phrasing like *"as used in a scientific publication"* demonstrably improves retrieval quality
- Domain-expert validation on every pair. Automated proxies are insufficient for this step
- Store as structured JSON: `{query, expected_chunks: [...], expected_answer, tier, module}`

Query tier examples:
- **API:** *"What arguments does `StencilProperties.from_accuracy` accept and what does each control, as used in a scientific computing context?"*
- **Workflow:** *"What is the sequence of calls to set up and solve a Poisson problem with pure Neumann boundary conditions?"*
- **Conceptual/math:** *"What is the role of the PHS kernel in the RBF-FD stencil assembly, and where is it implemented?"*

#### 0.2 — Eval Metrics

Scoreboard for every subsequent experiment:

1. **Retrieval recall@k** — does the correct source chunk appear in the top-k results? Measure k=3, k=5, k=10 separately.
2. **Faithfulness (RAGAS)** — once generation is wired in, does the output stay grounded in retrieved context or hallucinate?
3. **Constrained execution pass rate** — described in Phase 5; the harness design is specified now so it is ready when generation is wired.

---

### Phase 1 — Foundation

#### 1.1 — ChromaDB to Qdrant

**DONE.** Migration complete. Every point verified against the ChromaDB export (IDs, payload, metadata, vectors within 1e-5 tolerance). Dense ordered-ID parity confirmed at k=3, 5, 10. Hybrid parity confirmed — Qdrant hybrid matches 02b recall exactly.

**Open:** Manual BM25+RRF is still being mirrored from the notebook. Qdrant's native sparse + dense hybrid search replaces this entirely — evaluate and swap before treating the migration as production-complete.

#### 1.2 — Upgrade Embeddings

**Not started.** Comparison points to ablate:
- `voyage-code-2` (Voyage AI) — top of MTEB code leaderboard
- `Qwen3-Embedding-0.6B` — strong open-source alternative, good on code domain MTEB
- LLM-generated embeddings

Run the eval set after each swap. If recall@k doesn't improve, that's a result.

#### 1.3 — Enrich Chunk Metadata

**Not started.** Each chunk should carry `math_terms`, `cross_refs`, `chunk_type`, and `has_numba` in addition to current fields. Enables payload-filtered retrieval — when a query mentions "Laplacian," filter to chunks with that math term before vector search.

---

### Phase 2 — Retrieval Quality Improvements

#### 2.1 — Cross-encoder Reranking

**Implemented. Numbers in hand.**

`cross-encoder/ms-marco-MiniLM-L-6-v2` on top of Qdrant dense: recall@3 went 3/10 → 8/10 (+5).

**What's still open:**
- Reranker evaluated on dense-only Qdrant; needs evaluation on top of Qdrant hybrid
- Ranking quality for generation (chunk order matters for what the LLM sees) not yet measured
- Code-specific cross-encoder comparison not yet run

#### 2.2 — Hierarchical "Retrieve Small, Read Big"

**Not started.** Index at two granularities simultaneously. Retrieve fine (precision), expand to coarse before passing to the LLM (completeness).

#### 2.3 — Multi-representation Indexing

**Not started.** Attach multiple embeddings to the same chunk — code only, code + comments, comments only — using Qdrant named vector spaces. Ablation will isolate which representation contributes most.

#### 2.4 — The Trimodal Problem (KernelPack-Specific Research Gap)

**Not started.** A query like *"how do I set up a 4th-order RBF-FD Laplacian?"* requires three things simultaneously:

1. **Code:** `StencilProperties.from_accuracy(operator="lap", convergence_order=4, ...)`
2. **Math:** understanding that "4th order" means the polynomial augmentation degree and PHS exponent are linked
3. **Cross-reference:** knowing that `FDDiffOp` is the assembler that consumes those properties

No current RAG system handles this as a unified retrieval problem. Start with metadata filtering on `math_terms`, validate with the eval set, then move to dual-embedding retrieval.

*The citizen science RAG will hit this problem harder — a non-expert's query vocabulary will diverge furthest from codebase naming conventions.*

---

### Phase 3 — Add LLM Generation

Until this phase the system is retrieval-only. Phase 3 wires retrieval to generation.

#### 3.1 — Two Distinct Prompt Strategies

**For the coding agent (Project 1):**

```
You are assisting with code that uses the KernelPack library.
Use ONLY the following retrieved source context to answer.
Do not invent functions, classes, or arguments not present below.

[retrieved chunks]

Query: {query}
```

**For the citizen science interface (Project 2):**

```
A user wants to run a simulation. Their goal is: {natural_language_goal}

Based on the following KernelPack source context, outline the steps
in plain English first, then provide working Python code.

[retrieved chunks]
```

#### 3.2 — Adaptive Retrieval

A lightweight query classifier (a single LLM call) routes:
- Simple/direct → retrieve 3 chunks, single-hop
- Complex/workflow → retrieve 15 chunks across multiple modules, potentially multi-hop

The citizen science use case skews heavily toward complex/workflow queries.

---

### Phase 4 — MCP Wrapper

The RAG is exposed as MCP tools:

```
retrieve_code(query: str, module_filter?: str) -> list[CodeChunk]
explain_api(symbol_name: str) -> str
suggest_workflow(goal: str) -> WorkflowPlan
run_example(workflow_plan: WorkflowPlan) -> CodeSnippet
```

The MCP server is a thin wrapper — all real logic lives in the retrieval pipeline.

**For the coding agent (Project 1):** Codex calls `retrieve_code` before generating any KernelPack-related code. Enforced via system prompt. Verified via call logs — Codex should not answer from training data.

**For the citizen science interface (Project 2):** The user's natural language goal hits `suggest_workflow`, which returns a structured plan (geometry type, solver type, boundary condition type), which triggers `run_example` to produce starter code.

---

### Phase 5 — Constrained Execution Eval (Novel Contribution)

This is an evaluation approach that, to my knowledge, has not been done for scientific code RAG systems.

#### The Idea

A separate eval agent operates inside a sandbox where its only available context is what the RAG retrieved. The agent is only allowed to call functions that actually exist in KernelPack. The solution either works within KernelPack's real API surface or it fails, and the failure type is logged.

#### Implementation

1. **Sandbox:** Docker container with only `kernelpack-python` installed. No internet. No libraries except numpy/scipy.
2. **Tool interface:** Expose KernelPack's public API as explicit tool calls the eval agent can invoke.
3. **Failure taxonomy:**
   - `FunctionNotFound` → retrieval failure
   - `WrongSignature` → incomplete context
   - `WrongOutput` → reasoning failure, not retrieval failure
   - `CorrectOutput` → pass
4. **Attribution:** `FunctionNotFound` and `WrongSignature` failures are attributed back to which retrieval step failed.

**Why this is novel:** no existing eval framework for code RAG closes the loop between generation failure and retrieval failure attribution at this level of specificity. SWE-bench tells you if the patch worked. This tells you *why* the RAG failed to ground the patch correctly.

---

### Phase 6 — Expansion

Once KernelPack-python is working end-to-end:

1. **KernelPack-MATLAB:** Tree-sitter has a MATLAB grammar. Main challenge: MATLAB's documentation patterns differ; re-run the embedding evaluation.
2. **Larger codebases (Trilinos, PETSc, deal.ii):** The Meta-RAG pattern (summarize first, navigate to relevant subsystems, then retrieve) becomes necessary. The KernelPack architecture scales — summarization and navigation layers are additive.

---

## Full Research Map

| Problem | Status | Plan |
|---|---|---|
| Hybrid BM25 + dense retrieval | **Solved, migrated to Qdrant** | Evaluate Qdrant native sparse vectors to replace notebook-local BM25 |
| AST-aware chunking | **Solved** | Extend to hierarchical (Phase 2.2) |
| LLM summary enrichment | **Solved** — biggest recall lever found | Expand to all 6 modules; domain-expert validation pending |
| Cross-encoder reranking | **Implemented, numbers in hand** | Evaluate on hybrid (not dense-only); measure ranking quality for generation |
| Hierarchical "retrieve small, read big" | Proposed in research, not in production | Phase 2.2 |
| Code-specialized / LLM-generated embeddings | Active research area | Ablation in Phase 1.2 |
| Multi-representation indexing | Proposed, not in production | Phase 2.3 |
| Trimodal retrieval (code + math + papers) | Open research gap | Phase 2.4 |
| Generation grounding / trustworthy RAG | **Entirely untested** | Phase 3 |
| Adaptive retrieval | Proposed, not in production | Phase 3.2 |
| Constrained execution eval | Not done for scientific code RAG | Phase 5 — novel contribution |
| Failure attribution (retrieval → generation) | Open research gap | Phase 5 failure taxonomy |
| RAG for MATLAB scientific code | Barely explored | Phase 6 |
| Citizen science RAG | Architecture overlap with developer RAG in early phases | Diverges at generation + interface layer |

---

## Setup

Notebooks assume `kernelpack-python` is cloned as a sibling directory:

```bash
git clone https://github.com/ShankarLab/kernelpack-python
git clone https://github.com/<your-handle>/scientific-codebase-rag
```

Then open any notebook — the setup cell will confirm the path and install dependencies.
