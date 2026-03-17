import logging
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import end_text1

logger = logging.getLogger(__name__)


async def message_end_process(language, folder_name, output_filename=None):
    """
    Ends the process and generates a message in the specified language.
    Returns a message indicating the process has ended, translated as needed.
    Logs errors if required arguments are missing.
    """
    message = await compose_message(
        language,
        [end_text1],
        subfolder=folder_name + "/end_message" if folder_name else None,
    )

    return message
