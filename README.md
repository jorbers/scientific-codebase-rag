# Scientific Codebase RAG

> Retrieval-Augmented Generation for scientific computing libraries.

Scientific code carries a **trimodal retrieval problem** — *code*, *mathematics*,
and *cross-references* must be resolved simultaneously. A user asking
*"how do I set up a 4th-order RBF-FD Laplacian?"* needs the right function
signature, the mathematical relationship between convergence order and polynomial
degree, and awareness that a separate assembler class consumes those properties.
Standard dense retrieval fails here because the user's vocabulary and the
codebase's naming conventions are disjoint — and no amount of embedding tuning
bridges that gap alone.

The primary testbed is **kernelpack-python** — a Python library for RBF-FD
(radial basis function finite difference) numerical methods. It is representative
of the class of scientific libraries this research targets: domain-specific
mathematical vocabulary, naming conventions orthogonal to natural language, and a
user community that knows what they want numerically but not the library's API.

---

## Projects

### codebase-rag — Developer / Coding Agent RAG

A RAG layer for AI coding agents working with scientific codebases, exposed via
an MCP server. The core constraint: no hallucinated API calls. Every generated
function, class, and argument must exist in the actual codebase.

Active implementation. See [`codebase-rag/README.md`](codebase-rag/README.md)
for architecture and [`codebase-rag/SETUP.md`](codebase-rag/SETUP.md) to get
running.

### citizen-science-rag

Planned counterpart targeting non-expert users. **No implementation yet** —
`citizen-science-rag/` is a stub.

---

## Repo Structure

```
scientific-codebase-rag/
├── codebase-rag/        ← active implementation, MCP server, test suite
├── citizen-science-rag/ ← stub, no implementation
├── experiments/         ← shared retrieval experimentation (notebooks 01–04)
└── qdrant_storage/      ← local Qdrant persistence (gitignored)
```

---

## Experiments

Notebooks 01–04 in `experiments/` cover chunking strategy, embedding model
selection, BM25 tokenization, hybrid retrieval construction, ChromaDB → Qdrant
migration, and a generation baseline. Full descriptions and authoritative recall
numbers are in [`experiments/README.md`](experiments/README.md).
