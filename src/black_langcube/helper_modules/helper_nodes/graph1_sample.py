"""
sample nodes - one conditional node and one action node

any other node may be defined in a similar way to provide custom logic/functionality
"""

import logging
logger = logging.getLogger(__name__)

from typing import Annotated, Literal


# ----------------------------------
#  - define decision making about translation based on "language"
# ----------------------------------
def route_translatequestion_or_question2translation(state) -> Literal["translate_question", "question2translation"]:
    logger.info("----- route_translatequestion_or_question2translation -----")
    # If the language starts with "English", we assume no translation is needed
    # and we route to Question2Translation, otherwise we route to TranslateQuestionNode
    if state["language"].startswith("English"):
        return ["question2translation"] 
    else: 
        return ["translate_question"]

# ----------------------------------
# - pass the question as the translation
# ----------------------------------
class Question2Translation:
    def __init__(self, state, config):
        self.state = state
        self.config = config
        self.logger = logging.getLogger(__name__)

    def execute(self, extra_input=None):
        self.logger.info("----- Executing Question2Translation -----")
        # Here we assume that the question is already in the desired format
        # and we simply return it as the translation.
        result = self.state["question"]
        self.state["question_translation"] = result
        self.logger.info("----- Question2Translation execution completed -----")
        return {"question_translation": result}


    
