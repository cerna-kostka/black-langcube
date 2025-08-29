"""
TranslatorUsrNode is a subclass of LLMNode designed to handle translation tasks using a language model.
It utilizes a prompt template and output parser to generate translations based on user input and a specified target language.
Attributes:
    state (dict): The current state containing translation input and target language.
    config (dict): Configuration parameters for the node.
Methods:
    __init__(state, config):
        Initializes the TranslatorUsrNode with the given state and configuration.
    generate_messages():
        Constructs and returns a list of messages for the translation prompt, including a system message
        formatted with the target language and a human message containing the input text to be translated.
    execute(extra_input=None):
        Executes the translation process by updating the state with new input, running the language model chain,
        and returning the translation output, token usage, and target language.
"""

import logging
logger = logging.getLogger(__name__)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from black_langcube.llm_modules.llm_model import llm_low
from black_langcube.llm_modules.LLMNodes.LLMNode import LLMNode
from black_langcube.prompts.prompts import translator_usr

class TranslatorUsrNode(LLMNode):
    def __init__(self, state, config):
        super().__init__(state, config)

    def generate_messages(self):
        """
        Generate messages for translation.
        """
        messages = [
            ("system", translator_usr["system"].format(language=self.state.get("language", "English"))),
            ("human", self.state.get("translation_input", "")),
        ]
        return messages

    def execute(self, extra_input=None):
        self.logger.info("----- Executing TranslatorUsrNode -----")
        self.state["translation_input"] = extra_input.get("translation_input")
        self.state["language"] = extra_input.get("language")
        result, tokens = self.run_chain(extra_input=extra_input)
        
        self.logger.info("----- TranslatorUsrNode execution completed -----")

        return {"translation_output": result, "translation_tokens": tokens, "language": self.state.get("language")}
