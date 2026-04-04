# Add Parallel Fan-Out (Scatter-Gather) Pattern Support to Graph Workflows

## Summary

`black_langcube` currently only supports strictly sequential node execution within graphs and pipelines. This issue adds a **parallel fan-out** (scatter-gather) capability, allowing a single graph node to dispatch work to multiple downstream nodes simultaneously and collect their results before continuing.

## Goals / Objectives

- Enable a graph node to fan out to N downstream nodes that execute **concurrently**, not sequentially.
- Enable a fan-in (merge) step that waits for all parallel branches and aggregates their results into shared state.
- Allow the complete pipeline (`run_complete_pipeline`) to optionally execute independent graphs in parallel.
- Preserve backward compatibility — existing linear workflows must continue to work unchanged.

## Requirements

- Implement a `add_parallel_nodes` (or equivalent) helper on `BaseGraph` that wires a fan-out edge from one node to a list of nodes using LangGraph's `Send` API or parallel `add_edge` semantics.
- Implement a fan-in / merge step that collects outputs from all parallel branches into `GraphState` before the workflow proceeds.
- Alternatively (or additionally), provide an `asyncio.gather`-based utility in `process.py` for running independent top-level graphs concurrently.
- Update `LLMNode` and `robust_invoke_async` if needed to ensure they are safe to invoke concurrently.
- Provide at least one working example of a fan-out workflow (e.g., in `src/black_langcube/examples/`).
- Add unit and integration tests covering parallel dispatch and result merging.
- Update `README.md` and relevant docstrings to document the new pattern.

## Out of Scope

- Distributed / multi-process parallelism (e.g., Ray, Celery) — this issue targets in-process `asyncio` concurrency only.
- Dynamic fan-out where the number of branches is determined at runtime by LLM output (can be a follow-up issue).
- Changing the existing sequential API or graph classes.

## Additional Information

- LangGraph natively supports parallel branches via the [`Send` API](https://langchain-ai.github.io/langgraph/how-tos/branching/) and by adding multiple edges from one node to several target nodes — neither is currently used in this library.
- `asyncio.gather` can be used at the pipeline level to dispatch multiple `Graph.run()` coroutines simultaneously without any LangGraph changes.
- Care must be taken with shared mutable state in `GraphState` under concurrent access — consider using `Annotated` reducer fields (LangGraph convention) for fan-in aggregation.
- `robust_invoke_async` already uses `asyncio`/`await`; verify it is re-entrant before using it in concurrent contexts.

## Tasks

- **Research LangGraph fan-out primitives** — Study the `Send` API, parallel edges, and reducer-annotated state fields in LangGraph. Document which approach fits `BaseGraph`'s current design.
- **Design the fan-out API on `BaseGraph`** — Decide on the method signature (e.g., `add_parallel_nodes(source, [node_a, node_b], merge_node)`), state shape for collecting parallel results, and reducer strategy.
- **Implement fan-out on `BaseGraph`** — Add the fan-out wiring method to `graph_base.py`, ensuring sequential workflows are unaffected.
- **Implement fan-in / merge step** — Add a configurable merge node or built-in reducer that aggregates outputs from all parallel branches into `GraphState`.
- **Implement pipeline-level parallelism in `process.py`** — Add `run_parallel_pipeline` (or a flag on `run_complete_pipeline`) that `asyncio.gather`s independent graph runs.
- **Verify concurrency safety of `LLMNode` and `robust_invoke_async`** — Run concurrent invocations and confirm no race conditions or shared-state bugs.
- **Write example workflow** — Add `src/black_langcube/examples/parallel_fanout_workflow.py` demonstrating the pattern end-to-end.
- **Write unit and integration tests** — Cover: fan-out dispatch, fan-in merging, error in one branch, pipeline-level concurrency.
- **Update documentation** — Add section to `README.md` explaining the fan-out pattern with a code snippet.

## Acceptance Criteria

- [ ] `BaseGraph` exposes a method for wiring a fan-out from one node to multiple parallel nodes with a subsequent merge step.
- [ ] Parallel branches in a single graph execute concurrently (verified by timing or mock assertions), not sequentially.
- [ ] Fan-in step correctly aggregates results from all branches into `GraphState`.
- [ ] `run_complete_pipeline` (or a new variant) can execute independent graphs in parallel using `asyncio.gather`.
- [ ] All existing sequential workflows and tests continue to pass without modification.
- [ ] A working example file (`parallel_fanout_workflow.py`) is included in `src/black_langcube/examples/`.
- [ ] No magic values created
- [ ] Applied `ruff format .` and `ruff check --fix .`
- [ ] Pytest passed successfully
- [ ] Update project changelog

## Comments / Feedback

*Does the team prefer the LangGraph `Send`-based approach, `asyncio.gather` at the pipeline level, or both? -> Use the classic LangGraph fan-out/fan-in pattern using Send + a merge node. Any constraints on minimum LangGraph version should be noted here -> langgraph>=1.0*
