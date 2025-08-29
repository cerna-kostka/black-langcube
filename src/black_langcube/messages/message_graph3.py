import logging
logger = logging.getLogger(__name__)

from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import graph3_text1

def message_graph3(language, subfolder, output_filename):
    """
    Returns a message for graph3 workflow based on language.
    """
    message = compose_message(language, [graph3_text1], subfolder=subfolder / "message")

    return message