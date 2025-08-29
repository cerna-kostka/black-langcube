"""
This module defines the `TranslatorUsrSubgraf` class, a workflow subgraph for user-language translation tasks,
and its associated state class `TranslatorUsrState`. The subgraph leverages modular graph-based workflow
construction, integrating a translation node and managing state transitions.
Classes:
    TranslatorUsrState: Inherits from a dynamically loaded GraphState base class. Stores input, output, and token
        information for translation tasks.
        Attributes:
            translation_input (str): The input text to be translated.
            translation_output (str): The translated output text.
            translation_tokens (dict): Metadata or tokens related to the translation process.
    TranslatorUsrSubgraf: Inherits from a dynamically loaded BaseGraph class. Orchestrates the translation workflow
        by constructing a graph with a translation node and managing execution.
        Methods:
            __init__(config, subfolder=None): Initializes the subgraph with configuration and state.
            build_graph(): Constructs the workflow graph by adding nodes and edges.
            workflow_name (property): Returns the name of the workflow.
            run(extra_input=None): Executes the workflow, writes events to file, and returns a user message.
Dependencies:
    - langgraph.graph (START, END): For graph workflow control.
    - llm_modules.LLMNodes.subgraphs.translator_usr_subgraph.TranslatorUsrNode: The translation node.
    - messages.subgraphs.message_translator_usr: For generating user-facing messages.
    - helper_modules.get_basegraph_classes: Dynamically loads base graph and state classes.
    - logging: For workflow logging.
Note:
    The module avoids circular imports by dynamically loading base classes.
"""

import logging
logger = logging.getLogger(__name__)
from langgraph.graph import START, END

#from ..graph_base import BaseGraph, GraphState  # import removed to avoid circular import issues
from black_langcube.llm_modules.LLMNodes.subgraphs.translator_usr_subgraph import TranslatorUsrNode
from black_langcube.messages.subgraphs.message_translator_usr import message_translator_usr
from black_langcube.helper_modules.get_basegraph_classes import get_basegraph_classes


class TranslatorUsrState(get_basegraph_classes()[1]): # getting GraphState from BaseGraph
    translation_input: str
    translation_output: str
    translation_tokens: dict

class TranslatorUsrSubgraf(get_basegraph_classes()[0]): # getting BaseGraph from get_basegraph_classes
    def __init__(self, config, subfolder=None):
        super().__init__(TranslatorUsrState, user_message=None, folder_name=subfolder, language=None)
        self.state = TranslatorUsrState()
        self.config = config
        self.build_graph()
    
    def build_graph(self):
        # Instantiate the workflow nodes
        TranslatorUsrNode_instance = TranslatorUsrNode(self.state, self.config)
        # Add the TranslatorUsrNode to the graph
        self.add_node("translator_usr", TranslatorUsrNode_instance.execute)

        # Add edges to connect the nodes
        self.add_edge(START, "translator_usr")
        self.add_edge("translator_usr", END)

    @property
    def workflow_name(self):
        return "translator_usr_subgraf"

    def run(self, extra_input=None):
        logger.info("--- Starting TranslatorUsrSubgraf workflow ---")
        events = self.graph_streaming(extra_input or {}, recursion_limit=10)
        subfolder = self.write_events_to_file(events, self.output_filename)
        message = message_translator_usr(self.state.get("language"), subfolder, self.output_filename)
        logger.info("--- TranslatorUsrSubgraf workflow completed ---")
        return message