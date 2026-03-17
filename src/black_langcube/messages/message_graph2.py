import logging
from black_langcube.helper_modules.get_result_from_graph_outputs import (
    get_result_from_graph_outputs_async,
)
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import graph2_text1, graph2_text2

logger = logging.getLogger(__name__)


async def message_graph2(language, subfolder, output_filename):
    """
    Returns a message for graph2 workflow based on the language and keywords.
    """
    # Retrieve keywords from the graph2 output using the translation keys.
    if language.startswith("English"):
        # If the language is English, we get the result with no translation
        keywords = await get_result_from_graph_outputs_async(
            "keyword2translation",
            "",
            "keywords_translation",
            "",
            subfolder,
            output_filename,
        )
    else:
        # If the language is not English, we need to get the result from translation
        keywords = await get_result_from_graph_outputs_async(
            "translate_keyword",
            "",
            "keywords_translation",
            "",
            subfolder,
            output_filename,
        )

    message = await compose_message(
        language,
        [graph2_text1, keywords, graph2_text2],
        subfolder=subfolder / "message",
    )

    return message
