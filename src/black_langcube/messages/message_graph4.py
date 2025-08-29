import logging
logger = logging.getLogger(__name__)

import json

from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs
from black_langcube.messages.compose_message import compose_message
from black_langcube.messages.texts_sample import graph4_text_stop, graph4_text_continue


def message_graph4(language, subfolder, output_filename):
    """
    Returns a message for graph4 workflow based on recommendation and language.
    """
    # Retrieve recommendation from graph4 output.
    try:
        recommendation = get_result_from_graph_outputs(
            "number_searches_available",
            "",
            "recommendation",
            "",
            subfolder,
            output_filename
        )
        if not recommendation:
            logger.error("Recommendation not found in graph4 output")
            raise ValueError("Recommendation not found in graph4 output")
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Error getting recommendation value from graph4 output")
        raise RuntimeError("Error getting recommendation value from graph4 output") from e

    # Build message based on recommendation and language.
    if recommendation == "stop":
        message = compose_message(language, [graph4_text_stop], subfolder=subfolder / "message")
        return message
    elif recommendation == "continue":
        message = compose_message(language, [graph4_text_continue], subfolder=subfolder / "message")
        return message
    else:
        logger.error(f"Unknown recommendation value: {recommendation}")
        return "Unknown error occurred when assessing amount of sources. Please contact support."
