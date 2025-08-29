import logging
logger = logging.getLogger(__name__)

from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import graph2_text1, graph2_text2

def message_graph2(language, subfolder, output_filename):
    """
    Returns a message for graph2 workflow based on the language and keywords.
    """
    # Retrieve keywords from the graph2 output using the translation keys.
    if language.startswith("English"):
        # If the language is English, we get the result with no translation
        keywords = get_result_from_graph_outputs(
            "keyword2translation",
            "",
            "keywords_translation",
            "",
            subfolder,
            output_filename
        )
    else:
        # If the language is not English, we need to get the result from translation
        keywords = get_result_from_graph_outputs(
            "translate_keyword",
            "",
            "keywords_translation",
            "",
            subfolder,
            output_filename
        )

    message = compose_message(language, [graph2_text1, keywords, graph2_text2], subfolder=subfolder / "message")

    return message