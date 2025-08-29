"""
Module: message_translator_subgraf
This module defines the MessageTranslatorSubgraf class, which implements a subgraph workflow for translating messages using a language model. It leverages a graph-based workflow system to manage the translation process, including input handling, translation execution, and output management.
Classes:
    MessageTranslatorState: Inherits from GraphState (dynamically imported). Holds the state for the translation process, including input text, output text, and token information.
    MessageTranslatorSubgraf: Inherits from BaseGraph (dynamically imported). Orchestrates the translation workflow by building a graph with translation nodes, managing state, and handling execution and output.
Key Methods:
    __init__(self, config, subfolder=None): Initializes the subgraph with configuration and state.
    build_graph(self): Constructs the workflow graph by adding translation nodes and connecting edges.
    workflow_name (property): Returns the name of the workflow.
    run(self, extra_input=None): Executes the workflow, streams events, writes outputs, and returns the final translated message.
Dependencies:
    - langgraph.graph for graph workflow management.
    - llm_modules.LLMNodes.subgraphs.translator_usr_subgraph for translation node.
    - messages.subgraphs.message_message_translator for message formatting.
    - helper_modules.get_basegraph_classes for dynamic class imports.
    - logging for workflow tracing.
Usage:
    Instantiate MessageTranslatorSubgraf with configuration and optional subfolder, then call the run() method with input to perform message translation.

Note:
    The module avoids circular imports by dynamically loading base classes.
"""

import logging
from langgraph.graph import START, END

#from ..graph_base import BaseGraph, GraphState  # import removed to avoid circular import issues
from black_langcube.llm_modules.LLMNodes.subgraphs.translator_usr_subgraph import TranslatorUsrNode
from black_langcube.messages.subgraphs.message_message_translator import message_message_translator
from black_langcube.helper_modules.get_basegraph_classes import get_basegraph_classes

logger = logging.getLogger(__name__)

class MessageTranslatorState(get_basegraph_classes()[1]): # getting GraphState from BaseGraph
    translation_input: str
    translation_output: str
    translation_tokens: dict

class MessageTranslatorSubgraf(get_basegraph_classes()[0]): # getting BaseGraph from get_basegraph_classes
    def __init__(self, config, subfolder=None):
        super().__init__(MessageTranslatorState, user_message=None, folder_name=subfolder, language=None)
        self.state = MessageTranslatorState()
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
        return "message_translator_subgraf"

    def run(self, extra_input=None):
        logger.info("--- Starting MessageTranslatorSubgraf workflow ---")
        events = self.graph_streaming(extra_input or {}, recursion_limit=10)
        subfolder = self.write_events_to_file(events, self.output_filename)
        message = message_message_translator(self.state.get("language"), subfolder, self.output_filename)
        logger.info("--- MessageTranslatorSubgraf workflow completed ---")
        return message