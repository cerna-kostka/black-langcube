"""
This module defines the TranslatorEnSubgraf class, a workflow component for handling English translation tasks
within a graph-based architecture. It leverages dynamic base class retrieval to avoid circular imports and
integrates with LLM-based translation nodes.

Classes:
    TranslatorEngState: Inherits from a dynamically retrieved GraphState base class. Represents the state of the
        translation workflow, including input, output, tokens, and the original question.
    TranslatorEnSubgraf: Inherits from a dynamically retrieved BaseGraph class. Manages the construction and
        execution of the English translation subgraph workflow.

Key Methods:
    __init__(self, config, subfolder=None): Initializes the subgraph with configuration and state.
    build_graph(self): Constructs the workflow graph by adding translation nodes and edges.
    workflow_name(self): Returns the name of the workflow.
    run(self, extra_input=None): Executes the workflow, processes events, writes output, and returns a message.

Dependencies:
    - langgraph.graph for graph workflow management.
    - llm_modules.LLMNodes.subgraphs.translator_en_subgraph for the translation node.
    - messages.subgraphs.message_translator_en for message formatting.
    - helper_modules.get_basegraph_classes for dynamic base class retrieval.
    - logging for workflow logging.
Note:
    The module avoids circular imports by dynamically loading base classes.
"""

import logging
logger = logging.getLogger(__name__)
from langgraph.graph import START, END

#from ..graph_base import BaseGraph, GraphState # import removed to avoid circular import issues
from black_langcube.llm_modules.LLMNodes.subgraphs.translator_en_subgraph import TranslatorEngNode
from black_langcube.messages.subgraphs.message_translator_en import message_translator_en
from black_langcube.helper_modules.get_basegraph_classes import get_basegraph_classes


class TranslatorEngState(get_basegraph_classes()[1]):  # getting GraphState from BaseGraph
    translation_input: str
    translation_output: str
    translation_tokens: dict
    question: str

class TranslatorEnSubgraf(get_basegraph_classes()[0]): # getting BaseGraph from get_basegraph_classes
    def __init__(self, config, subfolder=None):
        super().__init__(TranslatorEngState, user_message=None, folder_name=subfolder, language=None)
        self.state = TranslatorEngState()
        self.config = config
        self.build_graph()

    def build_graph(self):
        # Instantiate the workflow nodes
        TranslatorEngNode_instance = TranslatorEngNode(self.state, self.config)
        # Add the TranslatorEngNode to the graph
        self.add_node("translator_en", TranslatorEngNode_instance.execute)

        # Add edges to connect the nodes
        self.add_edge(START, "translator_en")

    @property
    def workflow_name(self):
        return "translator_en_subgraf"

    def run(self, extra_input = None):
        logger.info("--- Starting TranslatorEnSubgraf workflow ---")
        events = self.graph_streaming(extra_input or {}, recursion_limit=10)
        subfolder = self.write_events_to_file(events, self.output_filename)
        message = message_translator_en(self.state.get("language"), subfolder, self.output_filename)
        logger.info("--- TranslatorEnSubgraf workflow completed ---")
        return message