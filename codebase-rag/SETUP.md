# Setup

Steps to go from a fresh clone to a running RAG pipeline.

---

## Prerequisites

- **Python ≥ 3.10** (tested on 3.12)
- **Docker** — for Qdrant
- **OpenAI API key** — required for ingestion (LLM summarization); set in `.env`
- **KernelPack source** — a local clone of `kernelpack-python`; you need the path to its `src/kernelpack` directory

---

## 1. Install dependencies

```bash
cd codebase-rag
python3 -m venv .venv
source .venv/bin/activate

pip install qdrant-client openai mcp torch transformers sentence-transformers \
    tree-sitter tree-sitter-python numpy pytest
pip install -e .
```

> `pip install -e .` installs this package in editable mode so `python -m kernelpack_rag` works.

---

## 2. Set environment variables

Create a `.env` file inside `codebase-rag/` and fill in your values:

```bash
# .env
OPENAI_API_KEY=sk-...          # required for ingestion (LLM summarization)
KP_SRC=/absolute/path/to/kernelpack-python/src/kernelpack   # required for ingestion and integration tests
QDRANT_HOST=localhost          # optional, default shown
QDRANT_PORT=6333               # optional, default shown
```

The CLI (`python -m kernelpack_rag`) automatically loads `.env` via `python-dotenv`. The eval scripts (`eval_mcq.py`, `eval_mcq_agent.py`) do the same. No manual export needed.

---

## 3. Start Qdrant

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -v "$(pwd)/../qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:latest
```

This mounts the persistent storage directory at `../qdrant_storage` (sibling of `codebase-rag/`). If you want an ephemeral instance instead, drop the `-v` flag.

Verify it's up:
```bash
curl http://localhost:6333/healthz
```

---

## 4. Ingest KernelPack

Run the ingestion pipeline against the KernelPack source. The `--source` flag must point to the directory literally named `kernelpack` inside the repo:

```bash
python -m kernelpack_rag ingest \
  --source /absolute/path/to/kernelpack-python/src/kernelpack
```

This populates the default vector spaces: `ctx__jinacode`, `bm25_code`, and `math__qwen3`.

**To also ingest the RBF paper corpus** (enables `trimodal` and `two_leg` retrieval plans):

```bash
python -m kernelpack_rag ingest \
  --source /absolute/path/to/kernelpack-python/src/kernelpack \
  --papers kernelpack_rag/papers
```

**To populate additional embedding spaces** (e.g., Qwen3 context or summary vectors), pass them with `--spaces`. Example:

```bash
python -m kernelpack_rag ingest \
  --source /absolute/path/to/kernelpack-python/src/kernelpack \
  --spaces ctx__jinacode bm25_code math__qwen3 ctx__qwen3 summary__qwen3
```

> Ingestion always calls `OpenAI()` for summarization. `OPENAI_API_KEY` must be set even if you are only populating non-summary spaces. Subsequent re-runs are fast because summaries are disk-cached in `summaries_cache/`.

---

## 5. Generate LLM summaries (optional, separate script)

If you want to pre-generate the `llm_summary` field in isolation without running the full ingestion:

```bash
KP_SRC=/absolute/path/to/kernelpack-python/src/kernelpack \
python scripts/run_summarize.py
```

Summaries are cached in `summaries_cache/` keyed by content hash. Safe to re-run; already-cached chunks are skipped.

---

## 6. Start the MCP server

```bash
python -m kernelpack_rag mcp
```

The server starts a stdio-based MCP server and exposes `retrieve_code` as a tool. To use it with Claude Code or another MCP client, register the server in your MCP config:

```json
{
  "mcpServers": {
    "kernelpack-rag": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "kernelpack_rag", "mcp"],
      "cwd": "/absolute/path/to/codebase-rag"
    }
  }
}
```

Retrieval logs are written to `logs/retrieval.jsonl` by default. Override with `LOG_PATH`.

---

## 7. Verify the pipeline

After ingestion, run the verification gate:

```bash
python -m kernelpack_rag verify \
  --source /absolute/path/to/kernelpack-python/src/kernelpack \
  --qa-pairs ../experiments/qa_pairs/benchmark_qa.json
```

This checks:
1. **Collection invariants** — schema matches config, no missing primary vectors, all cross-references resolve
2. **Golden-set parity** — recall@5 must be ≥ 0.80 (baseline: 8/10) against the 10-query golden set

Exit code 0 = pass. Exit code 1 = fail; the report shows which gate failed.

---

## 8. Run the test suite

Unit tests (no Qdrant required):

```bash
pytest tests/ -q
```

Integration tests (requires Qdrant on `localhost:6333` and `KP_SRC` set):

```bash
KP_SRC=/absolute/path/to/kernelpack-python/src/kernelpack \
pytest tests/ --integration -q
```

---

## Quick sanity check

Once Qdrant is populated, run a query from the CLI:

```bash
python -m kernelpack_rag query "how to build an RBF-FD differentiation matrix"
```

Or with a specific plan:

```bash
python -m kernelpack_rag query "laplacian operator on scattered nodes" --plan trimodal
```
