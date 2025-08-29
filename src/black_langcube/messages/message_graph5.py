import logging
logger = logging.getLogger(__name__)

from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import graph5_text1

def message_graph5(language, subfolder, output_filename):
    """
    Returns a message for graph5 workflow based on language.
    """
    message = compose_message(language, [graph5_text1], subfolder=subfolder / "message")
    
    return message