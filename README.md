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
│   ├── 02_eval_baseline.ipynb
│   └── qa_pairs/
│       └── solvers_qa.json
│   └── metadata/
│       └── solvers_summaries.json
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
| 02 | [`02_eval_baseline`](experiments/02_eval_baseline.ipynb) | Eval harness and retrieval experiments v1→v3 against 10 solver Q&A pairs | Vocabulary gap is real but closable via LLM summary enrichment; the bottleneck is ranking — 8/10 targets land in top-10 but fall outside top-3; cross-encoder reranking is the highest-leverage next step |

---

## Where Things Stand

*This table is versioned — updated as the research unfolds.*

| Component | Current State | Problem |
|---|---|---|
| Chunking | Tree-sitter at function/class boundaries | Good, but no hierarchy — no "read big" after "retrieve small" |
| Vector store | ChromaDB | Fine for prototyping; migrating to Qdrant |
| Embeddings | UniXcoder (microsoft/unixcoder-base) | Code-aware but not math-aware |
| Search | BM25 + dense, RRF fusion | Solid — keep, migrate to Qdrant native hybrid |
| Reranking | None | Missing a critical quality layer; highest-leverage next step |
| LLM generation | None | Retrieval-only; not yet end-to-end |
| Eval | 10 Q&A pairs (solvers module only) | Narrow scope; pairs need expansion and domain-expert validation |
| Metadata | Minimal (file path, function name, class name) | Missing math context, cross-references, chunk type |

**Key finding from current experiments:** vocabulary gap is real but not the primary bottleneck. After LLM summary enrichment, 8/10 targets appear in the top-10 retrieved neighborhood. The bottleneck is ranking, not retrieval. A cross-encoder reranker is the highest-leverage next step.

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

### Phase 1 — Foundation Migration

#### 1.1 — ChromaDB to Qdrant

Qdrant gives:
- Named vector spaces (used for multi-representation indexing in Phase 2)
- Payload filtering (filter by module, chunk type, etc. before vector search)
- Native sparse + dense hybrid search with built-in RRF
- Production-grade HNSW index for scale (Trilinos, PETSc later)

The manual BM25 + RRF implementation in the current pipeline works but is not persisted, not calibrated, and not tunable. Qdrant collapses three moving parts into one tool call.

Migration plan: same 252 filtered chunks, same UniXcoder embeddings, same metadata. Verify hybrid search + RRF reproduces existing recall before adding anything new. This infrastructure is shared by both projects.

#### 1.2 — Upgrade Embeddings

The current direction: use an LLM to generate embeddings rather than a dedicated embedding model — the current state of the art for code-domain retrieval.

Comparison points to ablate:
- `voyage-code-2` (Voyage AI) — top of MTEB code leaderboard
- `Qwen3-Embedding-0.6B` — strong open-source alternative, competitive on code domain MTEB
- LLM-generated embeddings — recommended direction

**Why this matters for KernelPack specifically:** the codebase mixes Python code, mathematical terminology (PHS, RBF, Legendre, Laplacian, BDF3), and scientific docstrings. A general embedding collapses `"lap"` (Laplacian operator) and `"lap"` (loop iteration) into nearby vectors. A code-aware or LLM-generated embedding does not.

Run the eval set after each swap. If recall@k doesn't improve, that is a result.

#### 1.3 — Enrich Chunk Metadata

Each chunk stored in Qdrant should carry:

```json
{
  "text": "...",
  "chunk_type": "function | class | module_docstring | inline_comment",
  "module": "kernelpack.rbffd",
  "parent_class": "FDDiffOp",
  "function_name": "assemble_op",
  "math_terms": ["laplacian", "PHS", "stencil"],
  "cross_refs": ["RBFStencil", "StencilProperties", "DomainDescriptor"],
  "has_numba": true,
  "source_file": "src/kernelpack/rbffd/assembler.py",
  "line_range": [42, 87]
}
```

`math_terms` and `cross_refs` are the key additions. They enable payload-filtered retrieval — when a query mentions "Laplacian," filter to chunks with that math term before vector search. Codex can generate these from source at index time.

---

### Phase 2 — Retrieval Quality Improvements

#### 2.1 — Cross-encoder Reranking

**Highest-leverage next step based on eval findings.**

Two-stage approach:
1. Hybrid BM25 + dense retrieves top 30–50 candidates (fast, cheap)
2. Cross-encoder scores each candidate against the query in full context (slower, accurate)
3. Keep top 5–8 for generation

The eval baseline shows 9/10 targets already land in the top-10 window. A cross-encoder on that window would plausibly recover 7–8/10 recall@3.

Starting point: `cross-encoder/ms-marco-MiniLM-L-6-v2`. Worth comparing against a code-specific cross-encoder once the baseline is established.

#### 2.2 — Hierarchical "Retrieve Small, Read Big"

Addresses the chunking size paradox. Index KernelPack at two granularities simultaneously in Qdrant:

- **Fine chunks:** 3–5 line logical units within a function. Small embedding footprint = precise retrieval.
- **Coarse chunks:** full function or class block. Sent to the LLM for generation context.

Every fine chunk carries a pointer to its parent coarse chunk. Retrieve fine (precision), expand to coarse before passing to the LLM (completeness).

Natural hierarchy for KernelPack:
- Fine: individual numerical operation or parameter block
- Coarse: full method (e.g., full `assemble_op`)
- Module: class docstring + `__init__` summary

#### 2.3 — Multi-representation Indexing

Core idea: attach multiple embeddings to the same chunk so different representations of the same content are indexed separately. A natural-language comment and raw code say the same thing differently — embed them with models suited to each modality rather than forcing one embedding to cover both.

Three embeddings per chunk:
- Code only
- Code + comments
- Comments only

All three target the vocabulary gap between how a scientist phrases a query ("conservation of mass") and how the code expresses it ("divergence free"). Ablation will isolate which representation contributes most. This benefits directly from Qdrant's named vector spaces.

#### 2.4 — The Trimodal Problem (KernelPack-Specific Research Gap)

This is where KernelPack diverges from generic code RAG — and where there is genuine research novelty.

A query like *"how do I set up a 4th-order RBF-FD Laplacian?"* requires three things simultaneously:

1. **Code:** `StencilProperties.from_accuracy(operator="lap", convergence_order=4, ...)`
2. **Math:** understanding that "4th order" means the polynomial augmentation degree and PHS exponent are linked
3. **Cross-reference:** knowing that `FDDiffOp` is the assembler that consumes those properties

No current RAG system handles this as a unified retrieval problem. The approach:

- At index time: run each function's docstring through an LLM to extract a "math context" summary. Store as metadata.
- At query time: if the query contains mathematical terminology, retrieve against both code embeddings and math context embeddings, then merge via two separate Qdrant vector spaces in one collection.

Start simple — metadata filtering on `math_terms` — validate with the eval set, then move to dual-embedding retrieval.

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

The key constraint: *"do not invent functions not present below."* This is the prompt-level version of constrained execution.

**For the citizen science interface (Project 2):**

```
A user wants to run a simulation. Their goal is: {natural_language_goal}

Based on the following KernelPack source context, outline the steps
in plain English first, then provide working Python code.

[retrieved chunks]
```

The citizen science prompt requires more scaffolding — the user does not know what a solver or domain descriptor is. The retrieved context has to bridge that gap. This is a harder generation problem than the coding agent case.

#### 3.2 — Adaptive Retrieval

Not every query needs heavy retrieval. A lightweight query classifier (a single LLM call) routes:
- Simple/direct → retrieve 3 chunks, single-hop
- Complex/workflow → retrieve 15 chunks across multiple modules, potentially multi-hop

The citizen science use case skews heavily toward complex/workflow queries — the classifier must handle natural-language goal descriptions, not just API questions.

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

**For the citizen science interface (Project 2):** The user's natural language goal hits `suggest_workflow`, which returns a structured plan (geometry type, solver type, boundary condition type), which triggers `run_example` to produce starter code. This is the higher-level tool the public-facing MCP server exposes: `plan_simulation(natural_language_goal)`.

---

### Phase 5 — Constrained Execution Eval (Novel Contribution)

This is an evaluation approach that, to my knowledge, has not been done for scientific code RAG systems.

#### The Idea

A separate eval agent operates inside a sandbox where its only available context is what the RAG retrieved — the retrieved chunks are its world. The agent is only allowed to call functions that actually exist in KernelPack. The solution either works within KernelPack's real API surface or it fails, and the failure type is logged.

This removes the need for human evaluation of every output. The sandbox is the judge.

#### Implementation

1. **Sandbox:** Docker container with only `kernelpack-python` installed. No internet. No libraries except numpy/scipy (KernelPack's own dependencies).

2. **Tool interface:** Expose KernelPack's public API as explicit tool calls the eval agent can invoke. The agent solves each eval task by composing real tool calls — it cannot reach outside what the RAG gave it.

3. **Failure taxonomy — log every failure by type:**
   - `FunctionNotFound` → the RAG retrieved context that implied a non-existent function. Retrieval failure.
   - `WrongSignature` → the function exists but was called with incorrect arguments. The RAG gave incomplete context about the API.
   - `WrongOutput` → the function ran but produced incorrect results. Reasoning failure, not retrieval failure.
   - `CorrectOutput` → pass.

4. **Attribution:** `FunctionNotFound` and `WrongSignature` failures are attributed back to which retrieval step failed to surface the right chunk. This closes the loop between eval failure and RAG improvement.

**Why this is novel:** no existing eval framework for code RAG closes the loop between generation failure and retrieval failure attribution at this level of specificity. SWE-bench tells you if the patch worked. This tells you *why* the RAG failed to ground the patch correctly — without requiring human review of every output.

*The constrained execution eval applies to both projects, though the citizen science case may warrant an additional failure category: "correct code, wrong abstraction level."*

---

### Phase 6 — Expansion

Once KernelPack-python is working end-to-end:

1. **KernelPack-MATLAB:** The retrieval architecture is language-agnostic. Tree-sitter has a MATLAB grammar. Main challenge: MATLAB's documentation patterns differ; re-run the embedding evaluation.

2. **Larger codebases (Trilinos, PETSc, deal.ii):** Orders of magnitude larger than KernelPack. The Meta-RAG pattern (summarize first, navigate to relevant subsystems, then retrieve) becomes necessary. The architecture built for KernelPack scales to this — summarization and navigation layers are additive.

---

## Full Research Map

| Problem | Status | Plan |
|---|---|---|
| Hybrid BM25 + dense retrieval | [x] Solved | Keep — migrate to Qdrant native hybrid |
| AST-aware chunking | [x] Solved | Extend to hierarchical (Phase 2.2) |
| Cross-encoder reranking | Standard in research; not yet in this pipeline | Phase 2.1 — highest-leverage next step |
| Hierarchical "retrieve small, read big" | Proposed in research, rarely in production | Phase 2.2 |
| Code-specialized / LLM-generated embeddings | Active research area | Ablation in Phase 1.2 |
| Multi-representation indexing | Phase 2.3 | |
| Trimodal retrieval (code + math + papers) | Open research gap | Phase 2.4 |
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
