# Codebase RAG

> Coming as the pipeline matures past the shared experimentation phase.

This project builds a trustworthy RAG layer for AI coding agents working with scientific codebases. The design constraint: the agent must not hallucinate API calls. Every generated function, class, and argument must exist in the actual codebase.

The pipeline is exposed via an MCP server — AI coding agents call `retrieve_code` before generating any KernelPack-related code, enforced via system prompt so the agent cannot answer from training data alone.

See the top-level [research plan](../README.md) for the full phase-by-phase roadmap. This folder fills in starting at Phase 3.
