"""MCP server for KernelPack RAG.

Exposes retrieval as MCP tools. The embedder and Qdrant client are
initialized once at server startup via the lifespan context manager and
shared across all tool calls.

Run via:
    python -m kernelpack_rag mcp
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent

from kernelpack_rag.config import CODE_COLLECTION, make_client
from kernelpack_rag.embed.jinacode import JinaCodeEmbedder
from kernelpack_rag.qdrant_utils import _field_equals_filter
from kernelpack_rag.retrieve import CodeChunk, hybrid

_LOG_PATH = Path(os.environ.get("LOG_PATH", "logs/retrieval.jsonl"))


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[dict]:
    client = make_client()
    yield {"client": client, "embedder": None}


mcp = FastMCP("kernelpack-rag", lifespan=_lifespan)


@mcp.tool()
def retrieve_code(
    query: str,
    module_filter: str | None = None,
    k: int = 10,
    ctx: Context = None,
) -> list[CodeChunk]:
    """Search the KernelPack code index and return matching code chunks."""
    client = ctx.request_context.lifespan_context["client"]
    embedder = ctx.request_context.lifespan_context["embedder"]
    if embedder is None:
        embedder = JinaCodeEmbedder()
        ctx.request_context.lifespan_context["embedder"] = embedder
    query_filter = (
        _field_equals_filter("module", module_filter)
        if module_filter is not None
        else None
    )
    candidates = hybrid(
        query,
        client=client,
        collection=CODE_COLLECTION,
        embedder=embedder,
        k=k,
        query_filter=query_filter,
        query_id=str(uuid.uuid4()),
        log_path=_LOG_PATH,
    )
    return [c.to_code_chunk() for c in candidates]


@mcp.tool()
def explain_api(symbol_name: str) -> CallToolResult:
    """Explain a KernelPack API symbol (not yet implemented)."""
    return CallToolResult(
        content=[TextContent(
            type="text",
            text="explain_api not yet implemented — Phase 3 scope",
        )],
        isError=True,
    )


@mcp.tool()
def suggest_workflow(goal: str) -> CallToolResult:
    """Suggest a KernelPack workflow for a scientific goal (not yet implemented)."""
    return CallToolResult(
        content=[TextContent(
            type="text",
            text="suggest_workflow not yet implemented — Phase 3 scope",
        )],
        isError=True,
    )


@mcp.tool()
def run_example(workflow_plan: dict) -> CallToolResult:
    """Execute a KernelPack workflow plan (not yet implemented)."""
    return CallToolResult(
        content=[TextContent(
            type="text",
            text="run_example not yet implemented — Phase 3 scope",
        )],
        isError=True,
    )


def main() -> None:
    mcp.run()
