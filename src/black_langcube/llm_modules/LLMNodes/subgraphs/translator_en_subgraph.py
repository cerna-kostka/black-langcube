"""
This module defines the TranslatorEngNode class, a node for handling English translation tasks within an LLM-based workflow.
Classes:
    TranslatorEngNode: Inherits from LLMNode and is responsible for generating translation prompts and executing translation logic.
Methods:
    __init__(self, state, config):
        Initializes the TranslatorEngNode with the given state and configuration.
    generate_messages(self):
        Constructs and returns a list of messages for the translation prompt, including a system message specifying the target language and a human message containing the input text to translate.
    execute(self, extra_input=None):
        Executes the translation process by updating the node's state with the provided input, running the translation chain, and returning the translation output, token usage, and target language.
"""

import logging
from black_langcube.llm_modules.LLMNodes.LLMNode import LLMNode
from black_langcube.prompts.prompts import translator_eng

logger = logging.getLogger(__name__)


class TranslatorEngNode(LLMNode):
    def __init__(self, state, config):
        super().__init__(state, config)

    def generate_messages(self):
        """
        Create messages for translation.
        """

        return [
            (
                "system",
                translator_eng["system"].format(
                    language=self.state.get("language", "English")
                ),
            ),
            ("human", self.state.get("translation_input", "")),
        ]

    async def execute(self, extra_input=None):
        self.logger.debug("----- Executing TranslatorEngNode -----")
        self.state["translation_input"] = extra_input.get("translation_input")
        self.state["language"] = extra_input.get("language")
        result, tokens = await self.run_chain(extra_input=extra_input)

        self.logger.debug("----- TranslatorEngNode execution completed -----")

        return {
            "translation_output": result,
            "translation_tokens": tokens,
            "language": self.state.get("language"),
        }
