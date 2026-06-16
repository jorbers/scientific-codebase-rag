# Scientific Codebase RAG — Research Log

> Retrieval-Augmented Generation for scientific computing libraries.

This repository documents ongoing research into RAG systems purpose-built for scientific codebases. The work targets a failure mode common to existing systems: they treat scientific code like generic software.

Scientific code carries a **trimodal retrieval problem** — *code*, *mathematics*, and *cross-references* must be resolved simultaneously. A user asking *"how do I set up a 4th-order RBF-FD Laplacian?"* needs the right function signature, the mathematical relationship between convergence order and polynomial degree, and awareness that a separate assembler class consumes those properties. Standard dense retrieval fails here because the user's vocabulary and the codebase's naming conventions are disjoint — and no amount of embedding tuning bridges that gap alone.

The primary testbed is **[kernelpack-python](https://github.com/ShankarLab/kernelpack-python)** — a Python library for RBF-FD (radial basis function finite difference) numerical methods. It is representative of the class of scientific libraries this research targets: domain-specific mathematical vocabulary, naming conventions orthogonal to natural language, and a user community that knows what they want numerically but not the library's API.

---

## Two Projects, One Foundation

Two RAG systems are under development in parallel. They share retrieval infrastructure through the indexing and search layers, then diverge at generation and interface.

### Project 1 — Developer / Coding Agent RAG

A trustworthy RAG layer for AI coding agents working with scientific codebases. The core design constraint: the agent must not hallucinate API calls. Every generated function, class, and argument must exist in the actual codebase. This is enforced through constrained retrieval and evaluated via a sandboxed execution harness that attributes generation failures back to specific retrieval gaps.

**Primary user:** developers and researchers using AI coding assistants with scientific libraries.

### Project 2 — Citizen Science RAG

Scientific simulation has always had an unspoken prerequisite: you had to already speak the language. Knowing that "heat spreading through a material" is a diffusion equation, that "irregular boundary" means non-convex domain, that the right tool is an RBF-FD solver — none of that is obvious, and all of it has historically been the price of entry.

This project removes that prerequisite. A user who asks *"how do I simulate what happens when a nuclear reactor overheats?"* should get a working, grounded answer — not a wall of jargon, not hallucinated pseudocode, not a closed door.

**Primary user:** anyone curious enough to ask.

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
│   │   ├── chunking.py
│   │   └── eval_utils.py
│   └── reports/
│
├── codebase-rag/                    ← Project 1
│   └── README.md
│
└── citizen-science-rag/             ← Project 2
    └── README.md
```

---

## Notebooks

| # | Notebook | Investigates | Key Finding |
|---|---|---|---|
| 01 | `01_naive_to_hybrid` | Chunking strategy, embedding model selection, hybrid retrieval construction | Fixed-line chunking and general-purpose embeddings both fail on scientific code; AST-aware chunking + UniXcoder + BM25/RRF hybrid establishes the right foundation |
| 02a | `02a_eval_bm25_mangling` | Research artifact — preserves broken whitespace BM25 tokenization to isolate the sparse leg's contribution by comparison with 02b | Whitespace tokenization renders BM25 functionally dead; 40-point recall@3 gap from one incorrect tokenization function |
| 02b | `02b_eval_baseline` | Authoritative retrieval experiments with identifier-aware BM25 tokenization — eval harness, failure mode diagnosis, LLM summary enrichment | LLM summary enrichment is the single biggest recall lever found; bottleneck shifts from retrieval coverage to ranking after summaries are added |
| 03 | `03_qdrant_migration` | ChromaDB → Qdrant migration, point-by-point verification, hybrid parity confirmation, cross-encoder reranker preview | Migration verified clean; hybrid parity confirmed; reranker adds significant recall over dense-only retrieval |

---

## Key Research Gaps This Work Addresses

- **Vocabulary mismatch between scientists and codebases.** Natural language queries ("convergence order," "compactly supported weighting") do not share tokens with code identifiers (`from_accuracy`, `pu_patch_weight`). Standard dense retrieval cannot bridge this without explicit vocabulary anchoring.
- **Math context is absent from existing code RAG systems.** No current system indexes the mathematical meaning of a function alongside its implementation. This is the gap that makes conceptual-tier queries fail even when API and workflow queries succeed.
- **Paper-to-code linkage is unexplored.** Scientific codebases implement equations from papers. The paper corpus is a natural-language bridge between how scientists describe problems and how the code names its functions. Exploiting this bridge via shared math term tags across two retrieval legs is a novel contribution.
- **Generation failure attribution is coarse.** Existing eval frameworks (SWE-bench and others) report whether a generated solution works. They do not distinguish retrieval failures from reasoning failures. A sandboxed execution harness with a typed failure taxonomy closes this loop.

---

## Setup

Notebooks assume `kernelpack-python` is cloned as a sibling directory:

```bash
git clone https://github.com/ShankarLab/kernelpack-python
git clone https://github.com/<your-handle>/scientific-codebase-rag
```

Then open any notebook — the setup cell will confirm the path and install dependencies.