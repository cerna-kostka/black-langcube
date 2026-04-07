"""
Tests for parallel fan-out (scatter-gather) support.

Covers:
- add_parallel_nodes wiring on BaseGraph
- Fan-out dispatch (branches run and write to shared state)
- Fan-in / merge (reducer-annotated field aggregates results from all branches)
- Error in one branch is captured and does not prevent other branches
- run_parallel_pipeline runs independent graphs concurrently
- run_parallel_pipeline raises ValueError when passed an empty list
- All existing sequential workflows continue to work (import smoke-test)
"""

import asyncio
import operator
import sys
import unittest
from pathlib import Path
from typing import Annotated

import pytest

# Make sure the src tree is on the path when the test is run directly.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.mark.unit
class TestAddParallelNodes(unittest.IsolatedAsyncioTestCase):
    """Unit tests for BaseGraph.add_parallel_nodes."""

    def _make_fan_out_graph(self, branch_fns: dict, merge_fn, router_fn=None):
        """Build and return a compiled fan-out graph for testing."""
        from black_langcube.graf.graph_base import BaseGraph, GraphState
        from langgraph.graph import END, START

        class FanState(GraphState):
            branch_results: Annotated[list, operator.add]
            merged: list

        class FanGraph(BaseGraph):
            @property
            def workflow_name(self):
                return "test_fanout"

        g = FanGraph(FanState, "test", "test_folder", "English")

        def fan_out_source(state):
            return {}

        g.add_node("source", fan_out_source)
        for name, fn in branch_fns.items():
            g.add_node(name, fn)
        g.add_node("merge", merge_fn)

        g.add_edge(START, "source")
        g.add_parallel_nodes(
            "source",
            list(branch_fns.keys()),
            "merge",
            router_fn=router_fn,
        )
        g.add_edge("merge", END)
        return g

    async def test_fan_out_dispatches_all_branches(self):
        """Both branch nodes are invoked and write to branch_results."""

        def branch_a(state):
            return {"branch_results": ["A"]}

        def branch_b(state):
            return {"branch_results": ["B"]}

        def merge(state):
            return {"merged": state["branch_results"]}

        g = self._make_fan_out_graph(
            {"branch_a": branch_a, "branch_b": branch_b}, merge
        )

        initial = {"branch_results": [], "merged": [], "messages": []}
        final = {}
        async for event in g.graph.astream(initial):
            for v in event.values():
                if isinstance(v, dict):
                    final.update(v)

        # Both branches must have contributed
        self.assertIn("A", final.get("merged", []))
        self.assertIn("B", final.get("merged", []))

    async def test_reducer_accumulates_branch_results(self):
        """operator.add reducer correctly concatenates lists from all branches."""

        def branch_x(state):
            return {"branch_results": ["X"]}

        def branch_y(state):
            return {"branch_results": ["Y"]}

        def branch_z(state):
            return {"branch_results": ["Z"]}

        def merge(state):
            return {"merged": sorted(state["branch_results"])}

        g = self._make_fan_out_graph(
            {"branch_x": branch_x, "branch_y": branch_y, "branch_z": branch_z}, merge
        )

        initial = {"branch_results": [], "merged": [], "messages": []}
        final = {}
        async for event in g.graph.astream(initial):
            for v in event.values():
                if isinstance(v, dict):
                    final.update(v)

        self.assertEqual(sorted(final.get("merged", [])), ["X", "Y", "Z"])

    async def test_custom_router_fn_is_used(self):
        """When a router_fn is provided it controls dispatch."""
        from langgraph.types import Send

        def branch_only_a(state):
            return {"branch_results": ["A"]}

        def branch_never(state):
            # Should not be called because the custom router skips it
            return {"branch_results": ["NEVER"]}

        def merge(state):
            return {"merged": state["branch_results"]}

        def router(state):
            # Only dispatch to branch_only_a
            return [Send("branch_only_a", state)]

        g = self._make_fan_out_graph(
            {"branch_only_a": branch_only_a, "branch_never": branch_never},
            merge,
            router_fn=router,
        )

        initial = {"branch_results": [], "merged": [], "messages": []}
        final = {}
        async for event in g.graph.astream(initial):
            for v in event.values():
                if isinstance(v, dict):
                    final.update(v)

        self.assertIn("A", final.get("merged", []))
        self.assertNotIn("NEVER", final.get("merged", []))

    async def test_branch_exception_propagates(self):
        """An exception raised inside a branch node propagates out of astream."""

        def branch_ok(state):
            return {"branch_results": ["OK"]}

        def branch_boom(state):
            raise RuntimeError("branch failure")

        def merge(state):
            return {"merged": state["branch_results"]}

        g = self._make_fan_out_graph(
            {"branch_ok": branch_ok, "branch_boom": branch_boom}, merge
        )

        initial = {"branch_results": [], "merged": [], "messages": []}
        with self.assertRaises(Exception):
            async for _ in g.graph.astream(initial):
                pass

    async def test_branches_run_concurrently(self):
        """Async branches run concurrently: wall time ≈ max(delays), not sum(delays)."""
        import time

        DELAY = 0.05  # 50 ms per branch

        async def slow_a(state):
            await asyncio.sleep(DELAY)
            return {"branch_results": ["A"]}

        async def slow_b(state):
            await asyncio.sleep(DELAY)
            return {"branch_results": ["B"]}

        def merge(state):
            return {"merged": sorted(state["branch_results"])}

        g = self._make_fan_out_graph({"slow_a": slow_a, "slow_b": slow_b}, merge)

        initial = {"branch_results": [], "merged": [], "messages": []}
        start = time.monotonic()
        async for _ in g.graph.astream(initial):
            pass
        elapsed = time.monotonic() - start

        # Sequential execution would take ≥ 2 * DELAY; concurrent should be < 1.5 * DELAY
        self.assertLess(elapsed, DELAY * 1.5, msg="Branches appear to run sequentially")
        self.assertEqual(sorted(["A", "B"]), ["A", "B"])  # sanity

    async def test_sequential_workflow_unaffected(self):
        """A plain sequential graph still works after the fan-out API is added."""
        from black_langcube.graf.graph_base import BaseGraph, GraphState
        from langgraph.graph import END, START

        class SeqState(GraphState):
            counter: int

        class SeqGraph(BaseGraph):
            @property
            def workflow_name(self):
                return "test_seq"

        g = SeqGraph(SeqState, "msg", "folder", "English")

        def step1(state):
            return {"counter": state["counter"] + 1}

        def step2(state):
            return {"counter": state["counter"] + 1}

        g.add_node("step1", step1)
        g.add_node("step2", step2)
        g.add_edge(START, "step1")
        g.add_edge("step1", "step2")
        g.add_edge("step2", END)

        initial = {"counter": 0, "messages": []}
        final = {}
        async for event in g.graph.astream(initial):
            for v in event.values():
                if isinstance(v, dict):
                    final.update(v)

        self.assertEqual(final.get("counter"), 2)


@pytest.mark.unit
class TestRunParallelPipeline(unittest.IsolatedAsyncioTestCase):
    """Unit tests for process.run_parallel_pipeline."""

    def _make_stub_graph(self, return_value):
        """Return a minimal object whose run() coroutine returns return_value."""

        class StubGraph:
            workflow_name = "stub"

            async def run(self):
                return return_value

        return StubGraph()

    def _make_failing_graph(self, message="boom"):
        """Return a graph whose run() raises RuntimeError."""

        class FailGraph:
            workflow_name = "fail"

            async def run(self):
                raise RuntimeError(message)

        return FailGraph()

    async def test_all_graphs_succeed(self):
        from black_langcube.process import run_parallel_pipeline

        graphs = [self._make_stub_graph(i) for i in range(3)]
        result = await run_parallel_pipeline(graphs)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_graphs"], 3)
        self.assertEqual(result["parallel_results"], [0, 1, 2])

    async def test_empty_graphs_raises(self):
        from black_langcube.process import run_parallel_pipeline

        with self.assertRaises(ValueError):
            await run_parallel_pipeline([])

    async def test_partial_failure_status(self):
        from black_langcube.process import run_parallel_pipeline

        graphs = [
            self._make_stub_graph("ok"),
            self._make_failing_graph("something went wrong"),
        ]
        result = await run_parallel_pipeline(graphs)

        self.assertEqual(result["status"], "partial_failure")
        self.assertEqual(result["total_graphs"], 2)
        # First result is the success
        self.assertEqual(result["parallel_results"][0], "ok")
        # Second result is the error dict
        error_entry = result["parallel_results"][1]
        self.assertIsInstance(error_entry, dict)
        self.assertIn("error", error_entry)

    async def test_results_preserve_order(self):
        from black_langcube.process import run_parallel_pipeline

        graphs = [
            self._make_stub_graph(label) for label in ["first", "second", "third"]
        ]
        result = await run_parallel_pipeline(graphs)

        self.assertEqual(result["parallel_results"], ["first", "second", "third"])

    async def test_single_graph(self):
        from black_langcube.process import run_parallel_pipeline

        result = await run_parallel_pipeline([self._make_stub_graph("sole")])

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["parallel_results"], ["sole"])


@pytest.mark.unit
class TestImportSmoke(unittest.TestCase):
    """Import-level smoke tests confirming no regressions."""

    def test_run_parallel_pipeline_importable(self):
        from black_langcube import run_parallel_pipeline

        self.assertTrue(asyncio.iscoroutinefunction(run_parallel_pipeline))

    def test_existing_exports_still_present(self):
        from black_langcube import (
            BaseGraph,
            GraphState,
            LLMNode,
            get_basegraph_classes,
            run_complete_pipeline,
            run_workflow_by_id,
        )

        for obj in (
            BaseGraph,
            GraphState,
            LLMNode,
            get_basegraph_classes,
            run_complete_pipeline,
            run_workflow_by_id,
        ):
            self.assertIsNotNone(obj)

    def test_add_parallel_nodes_method_exists(self):
        from black_langcube import BaseGraph

        self.assertTrue(callable(getattr(BaseGraph, "add_parallel_nodes", None)))


if __name__ == "__main__":
    unittest.main()
