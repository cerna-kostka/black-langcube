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
logger = logging.getLogger(__name__)
from langgraph.graph import StateGraph
import json
import os
from pathlib import Path

from typing_extensions import TypedDict
from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs

from black_langcube.messages.message_graph1 import message_graph1
from black_langcube.messages.message_graph2 import message_graph2
from black_langcube.messages.message_graph3 import message_graph3
from black_langcube.messages.message_graph4 import message_graph4
from black_langcube.messages.message_graph5 import message_graph5


class BaseGraph:
    def __init__(self, state_cls, user_message, folder_name, language):
        self.workflow = StateGraph(state_cls)
        self.user_message = user_message
        self.folder_name = folder_name
        self.language = language
        self.output_filename = self.get_output_filename()

    def add_node(self, name, node_callable):
        self.workflow.add_node(name, node_callable)

    def add_edge(self, from_node, to_node):
        self.workflow.add_edge(from_node, to_node)

    def add_conditional_edges(self, from_node, condition_callable):
        self.workflow.add_conditional_edges(from_node, condition_callable)

    def compile(self):
        """
        Compiles the review subgraph.
        """
        try:
            return self.workflow.compile()
        except Exception as e:
            logger.critical("Error compiling review subgraph")
            raise RuntimeError("Error compiling review subgraph") from e
    
    @property
    def graph(self):
        """
        Returns the compiled graph.
        """
        if not hasattr(self, '_graph'):
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
        """
        return Path(self.folder_name) / self.workflow_name

    def intro_info_check(self):
        # Basic validation
        if not self.user_message:
            logger.error("user_message must not be empty")
            raise ValueError("user_message must not be empty")
        if not self.folder_name:
            logger.error("folder_name must not be empty")
            raise ValueError("folder_name must not be empty")

    def graph_streaming(self, initial_state: dict, recursion_limit: int = 10, extra_input=None):
        """
        Streams the graph with the given initial state and recursion limit.
        """
        try:
            return self.graph.stream(initial_state, {"recursion_limit": recursion_limit, **(extra_input or {})})
        except Exception as e:
            logger.critical(f"Error running workflow for {self.workflow_name}")
            raise RuntimeError(f"Error running workflow for {self.workflow_name}") from e
    
    def write_events_to_file(self, events, output_filename: str):
        
        # Ensure folder_name is a string by calling it if it's callable.
        folder = self.folder_name() if callable(self.folder_name) else self.folder_name
        subfolder = Path(folder) / self.workflow_name
        #subfolder = Path(self.folder_name) / self.workflow_name
        subfolder.mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(subfolder, output_filename)
        
        try:
            output_file = open(file_path, "a", encoding="utf-8")
        except OSError as e:
            logger.critical(f"Error creating or opening '{output_filename}' in {subfolder}")
            raise RuntimeError(f"Error creating or opening '{output_filename}' in {subfolder}") from e

        for s in events:
            logger.info("=== EVENT ===")
            try:
                output_file.write(json.dumps(s, ensure_ascii=False, default=self.custom_json_serializer) + "\n")
            except OSError as e:
                logger.critical(f"Error writing to '{output_filename}' in {subfolder}")
                raise RuntimeError(f"Error writing to '{output_filename}' in {subfolder}") from e
            logger.info("=============")

        try:
            output_file.close()
        except OSError as e:
            logger.error(f"Error closing '{output_filename}' in {subfolder}")
            raise RuntimeError(f"Error closing '{output_filename}' in {subfolder}") from e
        
        return subfolder

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
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Retrieve language value from the 'graph1' folder.
    def get_language(self):

        subfolder1 = Path(self.folder_name) / "graph1"
        try:
            language = get_result_from_graph_outputs(
                "is_language",
                "",
                "language",
                "",
                subfolder1,
                "graph1_output.json")
            
            if not language:
                logger.error("Language not found in graph1_output.json")
                raise ValueError("Language not found in graph1_output.json")
            
            return language
            
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Error getting language value from graph1_output.json")
            raise RuntimeError("Error getting language value from graph1_output.json") from e

    def get_message(self, language, subfolder, output_filename):
        func_name = "message_" + self.workflow_name  # e.g. "message_graph5"
        message_func = globals().get(func_name)
        if callable(message_func):
            return message_func(language, subfolder, output_filename)
        else:
            raise ValueError(f"Function {func_name} not found")


    def run_base(self, initial_state, output_filename):
        
        events = self.graph_streaming(initial_state, recursion_limit=10)
        subfolder = self.write_events_to_file(events, output_filename)
        language = self.get_language()
        message = self.get_message(language, subfolder, output_filename)

        return message
    
    def run(self, output_filename: str = None):
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