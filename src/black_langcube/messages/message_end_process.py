import logging
logger = logging.getLogger(__name__)

from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import end_text1

def message_end_process(language, folder_name, output_filename=None):
    """
    Ends the process and generates a message in the specified language.
    Returns a message indicating the process has ended, translated as needed.
    Logs errors if required arguments are missing.
    """
    message = compose_message(language, [end_text1], subfolder=folder_name+"/end_message")
    
    return message