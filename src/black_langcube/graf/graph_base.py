"""
This module defines the `BaseGraph` class, which provides a foundational interface for constructing,
compiling, and executing stateful workflow graphs using the `langgraph` library.
It includes methods for adding nodes and edges, compiling the workflow, streaming graph execution,
handling output serialization, and retrieving workflow-specific messages and language settings.
The class is designed to be subclassed for specific graph workflows, with customizable behavior
for running and naming workflows. Additionally, a `GraphState` TypedDict is provided to define
the expected state structure for graph execution.

Classes:
    BaseGraph: Base class for managing and executing stateful workflow graphs.
    GraphState: TypedDict specifying the structure of the graph state.

Key Methods in BaseGraph:
    - __init__: Initializes the graph with state class, user message, folder, and language.
    - add_node: Adds a node to the workflow graph.
    - add_edge: Adds an edge between nodes in the workflow graph.
    - add_conditional_edges: Adds conditional edges from a node based on a callable.
    - compile: Compiles the workflow graph.
    - graph_streaming: Streams the execution of the graph with a given state.
    - write_events_to_file: Serializes and writes graph events to a file.
    - save_events_to_database: Hook invoked when folder_name is None; override in subclasses.
    - custom_json_serializer: Helper for serializing non-standard objects to JSON.
    - get_language: Retrieves the language setting from a previous graph output.
    - get_message: Retrieves a workflow-specific message function.
    - run_base: Runs the base workflow and returns the resulting message.
    - run: Placeholder for subclass-specific execution logic.

Properties:
    - graph: Returns the compiled workflow graph.
    - workflow_name: Returns the name of the workflow (to be overridden in subclasses).

Exceptions are logged and re-raised as RuntimeError or ValueError where appropriate.
"""

import logging
import json
import os
from pathlib import Path

import aiofiles
from langgraph.graph import StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from black_langcube.helper_modules.get_result_from_graph_outputs import (
    get_result_from_graph_outputs_async,
)

# Import all message functions to make them available in globals()
# These imports are accessed dynamically via globals().get() in the get_message method
# DO NOT REMOVE - Required for dynamic function lookup in get_message()
from black_langcube.messages.message_graph1 import message_graph1  # noqa: F401
from black_langcube.messages.message_graph2 import message_graph2  # noqa: F401
from black_langcube.messages.message_graph3 import message_graph3  # noqa: F401
from black_langcube.messages.message_graph4 import message_graph4  # noqa: F401
from black_langcube.messages.message_graph5 import message_graph5  # noqa: F401

logger = logging.getLogger(__name__)


class BaseGraph:
    def __init__(
        self, state_cls, user_message, folder_name=None, language=None, session_id=None
    ):
        if not folder_name and not session_id:
            raise ValueError("Either folder_name or session_id must be provided")
        self.workflow = StateGraph(state_cls)
        self.user_message = user_message
        self.folder_name = folder_name
        self.language = language
        self.session_id = session_id
        self.output_filename = self.get_output_filename()

    def add_node(self, name, node_callable):
        self.workflow.add_node(name, node_callable)

    def add_edge(self, from_node, to_node):
        self.workflow.add_edge(from_node, to_node)

    def add_conditional_edges(self, from_node, condition_callable):
        self.workflow.add_conditional_edges(from_node, condition_callable)

    def add_parallel_nodes(self, source_node, branch_nodes, merge_node, router_fn=None):
        """Wire a fan-out from *source_node* to *branch_nodes* with a subsequent *merge_node*.

        Uses LangGraph's ``Send`` API so that all branches are dispatched
        concurrently.  After every branch finishes its node function, control
        flows to *merge_node* where results can be aggregated.

        Args:
            source_node (str): Name of the node that triggers the fan-out.
                Its node function must be added separately via ``add_node``.
            branch_nodes (list[str]): Names of the nodes to run in parallel.
                Each must already be added via ``add_node``.
            merge_node (str): Name of the node that collects results from all
                branches.  Must already be added via ``add_node``.
            router_fn (callable | None): Optional function ``(state) ->
                list[Send]`` that controls what state each branch receives.
                When *None*, every branch receives an unmodified copy of the
                current state (i.e. the full graph state is broadcast to all
                branches).

        Example::

            def fan_out_node(state):
                return {"items": state["items"]}

            def branch_a(state):
                return {"parallel_results": [f"A: {state['items']}"]}

            def branch_b(state):
                return {"parallel_results": [f"B: {state['items']}"]}

            def merge_node(state):
                return {"merged": state["parallel_results"]}

            self.add_node("fan_out", fan_out_node)
            self.add_node("branch_a", branch_a)
            self.add_node("branch_b", branch_b)
            self.add_node("merge", merge_node)
            self.add_edge(START, "fan_out")
            self.add_parallel_nodes("fan_out", ["branch_a", "branch_b"], "merge")
            self.add_edge("merge", END)
        """
        if router_fn is None:

            def _router(state):
                return [Send(branch, state) for branch in branch_nodes]

            router_fn = _router

        self.workflow.add_conditional_edges(source_node, router_fn)
        for branch in branch_nodes:
            self.workflow.add_edge(branch, merge_node)

    def compile(self):
        """
        Compiles the review subgraph.
        """
        try:
            return self.workflow.compile()
        except Exception as e:
            logger.error("Error compiling review subgraph")
            raise RuntimeError("Error compiling review subgraph") from e

    @property
    def graph(self):
        """
        Returns the compiled graph.
        """
        if not hasattr(self, "_graph"):
            self._graph = self.compile()
        return self._graph

    @property
    def workflow_name(self):
        # Override this method in subclasses to return a specific name
        return "base_graph"

    def get_output_filename(self):
        return f"{self.workflow_name}_output.json"

    def get_subfolder(self):
        """
        Returns the subfolder path where the output files will be stored.
        Returns None when folder_name is not set (database mode).
        """
        if not self.folder_name:
            return None
        return Path(self.folder_name) / self.workflow_name

    def intro_info_check(self):
        # Basic validation
        if not self.user_message:
            logger.error("user_message must not be empty")
            raise ValueError("user_message must not be empty")
        if not self.folder_name and not self.session_id:
            logger.error("Either folder_name or session_id must be provided")
            raise ValueError("Either folder_name or session_id must be provided")

    async def graph_streaming(
        self, initial_state: dict, recursion_limit: int = 10, extra_input=None
    ):
        """
        Streams the graph with the given initial state and recursion limit.
        Returns list of events.
        """
        try:
            events = []
            state_with_meta = {**initial_state, "graph_name": self.workflow_name}
            async for event in self.graph.astream(
                state_with_meta,
                {"recursion_limit": recursion_limit, **(extra_input or {})},
            ):
                events.append(event)
            return events
        except Exception as e:
            logger.error(f"Error running workflow for {self.workflow_name}")
            raise RuntimeError(
                f"Error running workflow for {self.workflow_name}"
            ) from e

    async def write_events_to_file(self, events, output_filename: str):
        """
        Serialize *events* and append them as JSON lines to *output_filename* inside
        the workflow sub-folder derived from ``folder_name``.

        When ``folder_name`` is not set (database-only mode), the method delegates to
        :meth:`save_events_to_database` and returns ``None`` without touching the
        filesystem.  Subclasses that require database persistence should override
        :meth:`save_events_to_database` rather than this method.
        """
        if not self.folder_name:
            await self.save_events_to_database(events, output_filename)
            return None
        # Ensure folder_name is a string by calling it if it's callable.
        folder = self.folder_name() if callable(self.folder_name) else self.folder_name
        subfolder = Path(folder) / self.workflow_name
        subfolder.mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(subfolder, output_filename)

        try:
            async with aiofiles.open(file_path, "a", encoding="utf-8") as output_file:
                for s in events:
                    logger.debug("=== EVENT ===")
                    try:
                        await output_file.write(
                            json.dumps(
                                s,
                                ensure_ascii=False,
                                default=self.custom_json_serializer,
                            )
                            + "\n"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error writing to '{output_filename}' in {subfolder}"
                        )
                        raise RuntimeError(
                            f"Error writing to '{output_filename}' in {subfolder}"
                        ) from e
                    logger.debug("=============")
        except OSError as e:
            logger.error(
                f"Error creating or opening '{output_filename}' in {subfolder}"
            )
            raise RuntimeError(
                f"Error creating or opening '{output_filename}' in {subfolder}"
            ) from e

        return subfolder

    async def save_events_to_database(self, events, output_filename: str):
        """
        Extension hook for persisting *events* when no filesystem folder is configured.

        This method is called by :meth:`write_events_to_file` whenever ``folder_name``
        is ``None`` (or otherwise falsy).  The default implementation is intentionally
        a no-op so that existing deployments that do not require database storage are
        unaffected.

        Override this method in a subclass to persist events to a database or any
        other storage backend::

            class MyGraph(BaseGraph):
                async def save_events_to_database(self, events, output_filename):
                    await db.store(self.session_id, output_filename, events)

        Args:
            events: The list of event dicts collected during graph streaming.
            output_filename: The logical file name that would have been used had
                a folder been configured (e.g. ``"graph1_output.json"``).
        """

    # to help with JSON serialization for HumanMessage
    def custom_json_serializer(obj, extra_input=None):
        """
        Custom JSON serializer for objects that are not serializable by default.
        """
        if hasattr(obj, "dict"):
            return obj.dict()
        if hasattr(obj, "content"):
            return {"type": obj.__class__.__name__, "content": obj.content}
        # Handle graph objects (like SearchCoreSubgraf) by returning a simplified representation.
        if hasattr(obj, "workflow_name"):
            return {"workflow_name": obj.workflow_name}
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    # Retrieve language value from the 'graph1' folder.
    async def get_language(self):

        subfolder1 = Path(self.folder_name) / "graph1"
        try:
            language = await get_result_from_graph_outputs_async(
                "is_language",
                "",
                "language",
                "",
                subfolder1,
                "graph1_output.json",
            )

            if not language:
                logger.error("Language not found in graph1_output.json")
                raise ValueError("Language not found in graph1_output.json")

            return language

        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error getting language value from graph1_output.json")
            raise RuntimeError(
                "Error getting language value from graph1_output.json"
            ) from e

    async def get_message(self, language, subfolder, output_filename):
        func_name = "message_" + self.workflow_name  # e.g. "message_graph5"
        message_func = globals().get(func_name)
        if callable(message_func):
            return await message_func(language, subfolder, output_filename)
        else:
            raise ValueError(f"Function {func_name} not found")

    async def run_base(self, initial_state, output_filename):
        events = await self.graph_streaming(initial_state, recursion_limit=10)
        subfolder = await self.write_events_to_file(events, output_filename)
        language = await self.get_language()
        message = await self.get_message(language, subfolder, output_filename)
        return message

    async def run(self, output_filename: str = None):
        """
        Override this method in subclasses to run the graph with specific parameters.
        If no output_filename is provided, it will use the default filename based on the workflow name.
        """
        return "Define run method in subclass to execute the graph."


class GraphState(TypedDict, total=False):
    messages: list
    question_translation: str
    folder_name: str
    language: str
    session_id: str
    graph_name: str
