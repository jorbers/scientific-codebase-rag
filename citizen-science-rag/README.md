# Citizen Science RAG

> Coming as the pipeline matures past the shared experimentation phase.

This project is the public-facing counterpart to the codebase RAG — a system that meets curious people where they are and hands them working code for whatever they want to simulate, no prerequisites required.

The pipeline is exposed via an MCP server with a single high-level tool: `plan_simulation(natural_language_goal)`. A user describes what they want to model in plain language — the MCP handles the translation to geometry type, solver type, and boundary conditions, then returns working starter code. The entire complexity of scientific simulation is abstracted behind one call.

See the top-level [research plan](../README.md) for the full phase-by-phase roadmap. This folder fills in starting at Phase 3.