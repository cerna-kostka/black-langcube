"""
LLMNode is a base class for nodes that interact with language models in a chain-of-thought or prompt-composition workflow.
Attributes:
    state (Any): The current state or context for the node.
    config (Any): Configuration parameters for the node.
    logger (logging.Logger): Logger instance for the node.
Methods:
    __init__(state, config):
        Initializes the LLMNode with the given state and configuration.
    generate_messages():
        Abstract method to return a list of messages for prompt composition.
        Must be implemented by subclasses.
    get_llm():
        Returns the language model instance to use. Can be overridden by subclasses.
    get_parser():
        Returns the output parser for the language model's response. Can be overridden by subclasses.
    run_chain(extra_input=None):
        Prepares and executes the chain using the generated messages, language model, and parser.
        Returns the result and token usage.
    execute(extra_input=None):
        Executes the chain and returns the result and token usage.
"""

import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from black_langcube.llm_modules.llm_model import default_llm
from black_langcube.llm_modules.robust_invoke import robust_invoke


class LLMNode:
    def __init__(self, state, config):
        self.state = state
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_messages(self):
        """Return a list of messages for prompt composition.
        Must be implemented by subclasses."""
        raise NotImplementedError

    def get_llm(self):
        """Return the appropriate language model. Subclasses can override this if needed."""
        return default_llm  # or appropriate llm instance

    def get_parser(self):
        """Return the output parser. Subclasses can override if needed."""
        return StrOutputParser()

    def run_chain(self, extra_input=None):
        """Prepare the chain with messages, LLM, and parser."""
        messages = self.generate_messages()
        prompt = ChatPromptTemplate.from_messages(messages)
        chain =  prompt | self.get_llm() | self.get_parser()
        result, tokens = robust_invoke(chain, extra_input)
        return result, tokens

    def execute(self, extra_input=None):
        result, tokens = self.run_chain(extra_input)
        return result, tokens
        