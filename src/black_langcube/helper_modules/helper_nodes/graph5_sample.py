"""
sample nodes - one node that imports data from previous graph outputs and user chosen language

any other node may be defined in a similar way to provide custom logic/functionality
"""

import logging
logger = logging.getLogger(__name__)

import json
from pathlib import Path
from langchain_core.runnables import RunnableConfig

from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs


# ----------------------------------
# Graph 5 Node 1 = Searches & User Chosen Language Import
# ----------------------------------
class ImportUtility:
    def __init__(self, state: dict, config: RunnableConfig):
        self.state = state
        self.config = config
        self.logger = logging.getLogger(__name__)

    def execute(self, extra_input=None):
        self.logger.info("----- Executing ImportUtility -----")
        # Here we would implement the logic for importing the necessary data
        subfolder_name = Path(self.state["folder_name"]) / "graph4"

        try:
            foo = get_result_from_graph_outputs("foo", "", "foo", "", subfolder_name, "graph4_output.json")
        except (OSError, json.JSONDecodeError) as e:
            self.logger.error("Error getting foo from graph4_output.json")
            raise RuntimeError("Error getting foo from graph4_output.json") from e
    
        try:
            bar = get_result_from_graph_outputs("bar", "", "baz", "", subfolder_name, "graph4_output.json")
        except (OSError, json.JSONDecodeError) as e:
            self.logger.error("Error getting bar from graph4_output.json")
            raise RuntimeError("Error getting bar from graph4_output.json") from e

        # Check if foo and bar are lists
        if not isinstance(foo, list):
            self.logger.error("foo is not a list")
            raise ValueError("foo is not a list")
        if not isinstance(bar, list):
            self.logger.error("bar is not a list")
            raise ValueError("bar is not a list")
        # Check if foo and bar are not empty
        if not foo:
            self.logger.warning("foo is empty, proceeding with an empty list")
            foo = []
        if not bar:
            self.logger.warning("bar is empty, proceeding with an empty list")
            bar = []

        try:
            user_language = get_result_from_graph_outputs("baz", "", "user_language", "", subfolder_name, "graph4_output.json")
        except (OSError, json.JSONDecodeError) as e:
            self.logger.error("Error getting user_language from graph4_output.json")
            raise RuntimeError("Error getting user_language from graph4_output.json") from e

        subfolder_name1 = Path(self.state["folder_name"]) / "graph1"
        try:
            if self.state["language"].startswith("English"):
                question_translation = get_result_from_graph_outputs("question", "", "question_translation", "", subfolder_name1, "graph1_output.json")
            else:
                question_translation = get_result_from_graph_outputs("translate_question", "", "question_translation", "", subfolder_name1, "graph1_output.json")
        except (OSError, json.JSONDecodeError) as e:
            self.logger.error("Error getting question_translation from graph1_output.json")
            raise RuntimeError("Error getting question_translation from graph1_output.json") from e

        self.logger.info("----- ImportUtility execution completed -----")

        self.state["foo"] = foo
        self.state["bar"] = bar
        self.state["user_language"] = user_language
        self.state["question_translation"] = question_translation

        return {
            "foo": foo,
            "bar": bar,
            "user_language": user_language,
            "question_translation": question_translation
    }