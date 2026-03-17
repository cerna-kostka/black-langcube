import logging
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import graph3_text1

logger = logging.getLogger(__name__)


async def message_graph3(language, subfolder, output_filename):
    """
    Returns a message for graph3 workflow based on language.
    """
    message = await compose_message(
        language, [graph3_text1], subfolder=subfolder / "message"
    )

    return message
