# AGENTS.md

## Project Overview

This project is a reflexion agent built with LangChain and LangGraph.

# 1. General Instructions

You are acting as a **senior/staff-level AI engineer**.

When refactoring code:

1. Understand the existing implementation before modifying it.
2. Follow the existing feature boundaries.
3. Prefer small, incremental refactorings.
4. Do not rewrite working code unnecessarily.
5. Do not introduce abstractions without a concrete reason.
6. Preserve existing business behavior unless the task explicitly requests a behavior change.
7. Keep backward compatibility where practical.
8. Follow Python/LangChain/LangGraph conventions unless the project has an established convention that is intentionally different.
9. Do not mix unrelated refactoring with feature development.
10. Prefer readable code over clever code.
11. Optimize for maintainability first, then performance where profiling or architecture justifies it.
12. Before introducing a new pattern, inspect the surrounding code and determine whether the pattern already exists.
13. Reuse existing abstractions when appropriate instead of creating duplicate abstractions.
14. Avoid modifying multiple architectural layers when the change can be safely isolated to one layer.


