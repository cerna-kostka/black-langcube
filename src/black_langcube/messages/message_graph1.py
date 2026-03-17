import logging
from black_langcube.helper_modules.get_result_from_graph_outputs import (
    get_result_from_graph_outputs_async,
)
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import graph1_text1, graph1_text2

logger = logging.getLogger(__name__)


async def message_graph1(language, subfolder, output_filename):
    """
    Returns a message for graph1 workflow based on the language and refining questions.
    """

    if language.startswith("English"):
        # If the language is English, we get the result with no translation
        questions_out = await get_result_from_graph_outputs_async(
            "refquestions2translation",
            "",
            "refining_question_usr",
            "",
            subfolder,
            output_filename,
        )
    else:
        # If the language is not English, we need to get the result from translation
        questions_out = await get_result_from_graph_outputs_async(
            "translate_refquestion",
            "",
            "refining_question_usr",
            "",
            subfolder,
            output_filename,
        )

    message = await compose_message(
        language,
        [graph1_text1, questions_out, graph1_text2],
        subfolder=subfolder / "message",
    )

    return message
