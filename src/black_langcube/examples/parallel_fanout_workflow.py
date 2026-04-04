"""
Parallel Fan-Out (Scatter-Gather) Example for Black LangCube

This example demonstrates the ``add_parallel_nodes`` helper on ``BaseGraph``
and the ``run_parallel_pipeline`` function in ``process.py``.

Two independent patterns are shown:

1. **Intra-graph fan-out** — a single LangGraph workflow fans out from one
   node to two branch nodes that run concurrently, then merges their results
   in a dedicated merge node.

2. **Pipeline-level parallelism** — two completely separate graph instances
   are launched simultaneously with ``run_parallel_pipeline``.

No real LLM calls are made; all nodes are pure Python functions so the
example runs without an OpenAI API key.
"""

import asyncio
import logging
import operator
from pathlib import Path
from typing import Annotated

from langgraph.graph import END, START

from black_langcube.graf.graph_base import BaseGraph, GraphState
from black_langcube.process import run_parallel_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Intra-graph fan-out
# ---------------------------------------------------------------------------


class FanOutState(GraphState):
    """State for the fan-out workflow.

    ``branch_results`` uses ``operator.add`` as its reducer so that each
    branch can append its own list without overwriting what other branches
    wrote.
    """

    topic: str
    branch_results: Annotated[list, operator.add]
    merged_summary: str


class ParallelFanOutWorkflow(BaseGraph):
    """
    A workflow with one fan-out node dispatching to two parallel branches.

    Graph topology::

        START → prepare → [branch_a, branch_b] (concurrent) → merge → END
    """

    def __init__(self, topic: str, folder_name: str, language: str = "English"):
        self._topic = topic
        super().__init__(FanOutState, topic, folder_name, language)
        self._build_graph()

    def _build_graph(self):
        def prepare(state: FanOutState) -> dict:
            """Set up the topic that branches will process."""
            logger.info("prepare: fanning out on topic '%s'", state["topic"])
            return {}

        def branch_a(state: FanOutState) -> dict:
            """Simulate work done by branch A (e.g. a search sub-task)."""
            result = f"[Branch A] analysed '{state['topic']}' from angle A"
            logger.info(result)
            return {"branch_results": [result]}

        def branch_b(state: FanOutState) -> dict:
            """Simulate work done by branch B (e.g. a translation sub-task)."""
            result = f"[Branch B] analysed '{state['topic']}' from angle B"
            logger.info(result)
            return {"branch_results": [result]}

        def merge(state: FanOutState) -> dict:
            """Aggregate results from both branches."""
            summary = " | ".join(state["branch_results"])
            logger.info("merge: %s", summary)
            return {"merged_summary": summary}

        self.add_node("prepare", prepare)
        self.add_node("branch_a", branch_a)
        self.add_node("branch_b", branch_b)
        self.add_node("merge", merge)

        self.add_edge(START, "prepare")
        # Wire the fan-out: prepare → {branch_a, branch_b} (concurrent) → merge
        self.add_parallel_nodes("prepare", ["branch_a", "branch_b"], "merge")
        self.add_edge("merge", END)

    @property
    def workflow_name(self) -> str:
        return "parallel_fanout"

    async def run(self, output_filename: str = None) -> str:
        initial_state: FanOutState = {
            "topic": self._topic,
            "branch_results": [],
            "merged_summary": "",
            "messages": [],
            "folder_name": self.folder_name,
            "language": self.language,
        }

        final_state: dict = {}
        async for event in self.graph.astream(initial_state):
            for node_output in event.values():
                if isinstance(node_output, dict):
                    final_state.update(node_output)

        return final_state.get("merged_summary", "")


# ---------------------------------------------------------------------------
# 2. Pipeline-level parallelism helper
# ---------------------------------------------------------------------------


class SimpleCounterState(GraphState):
    """Minimal state for the pipeline-level parallelism demo."""

    label: str
    count: int


class CounterWorkflow(BaseGraph):
    """A trivial workflow used to demonstrate ``run_parallel_pipeline``."""

    def __init__(self, label: str, folder_name: str):
        self._label = label
        super().__init__(SimpleCounterState, label, folder_name, "English")
        self._build_graph()

    def _build_graph(self):
        def count_node(state: SimpleCounterState) -> dict:
            result = state["count"] + 1
            logger.info("[%s] count → %d", state["label"], result)
            return {"count": result}

        self.add_node("count", count_node)
        self.add_edge(START, "count")
        self.add_edge("count", END)

    @property
    def workflow_name(self) -> str:
        return f"counter_{self._label}"

    async def run(self, output_filename: str = None) -> dict:
        initial_state: SimpleCounterState = {
            "label": self._label,
            "count": 0,
            "messages": [],
            "folder_name": self.folder_name,
            "language": "English",
        }

        final: dict = {}
        async for event in self.graph.astream(initial_state):
            for node_output in event.values():
                if isinstance(node_output, dict):
                    final.update(node_output)

        return {"label": self._label, "count": final.get("count", 0)}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def demo_intra_graph_fanout(output_folder: str):
    print("\n=== 1. Intra-graph fan-out ===")
    workflow = ParallelFanOutWorkflow(
        topic="Climate change mitigation",
        folder_name=output_folder,
    )
    result = await workflow.run()
    print(f"Merged summary: {result}")
    return result


async def demo_pipeline_parallelism(output_folder: str):
    print("\n=== 2. Pipeline-level parallelism ===")
    graphs = [
        CounterWorkflow("alpha", output_folder),
        CounterWorkflow("beta", output_folder),
        CounterWorkflow("gamma", output_folder),
    ]
    result = await run_parallel_pipeline(graphs)
    print(f"Status : {result['status']}")
    print(f"Results: {result['parallel_results']}")
    return result


async def main():
    output_folder = str(Path(__file__).parent / "example_output" / "parallel_fanout")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    summary = await demo_intra_graph_fanout(output_folder)
    assert summary, "Fan-out workflow returned an empty summary"

    pipeline = await demo_pipeline_parallelism(output_folder)
    assert pipeline["status"] == "completed"
    assert len(pipeline["parallel_results"]) == 3

    print("\nAll assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
