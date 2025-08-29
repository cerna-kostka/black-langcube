from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs


def message_message_translator(language, subfolder, output_filename):
    """
    Returns a message for graph1 workflow based on the language and refining questions.
    """
    translation_out = get_result_from_graph_outputs(
        "translator_usr",
        "",
        "translation_output",
        "",
        subfolder,
        output_filename
    )
    tokens_out = get_result_from_graph_outputs(
        "translator_usr",
        "",
        "translation_tokens",
        "",
        subfolder,
        output_filename
    )
    message = translation_out, tokens_out

    return message