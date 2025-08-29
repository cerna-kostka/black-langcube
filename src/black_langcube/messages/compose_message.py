import logging
logger = logging.getLogger(__name__)

from langchain_core.runnables import RunnableConfig
from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs

def compose_message(language, components, subfolder=None):
    """
    Compose a message for a specific language given a list of components.
    
    Each component can either be:
      - A tuple (english_text, czech_text) representing the text in English and Czech.
        For languages other than English or Czech, the English version is used as the base for translation.
      - A string (or any object that can be cast to a string) that is included as is.
      
    The components are joined by double newlines.
    
    Example usage:
      message = compose_message(language, [
          ("Here is a list of keywords - select the ones that are relevant to you:",
           "Zde je seznam klíčových slov - vyberte ta, která jsou pro vás relevantní:"),
          dynamic_content,
          ("After selecting all relevant items, submit your answer by pressing the button below.",
           "Po výběru všech relevantních položek svou odpověď odešlete stiskem tlačítka níže.")
      ])
    """

    from black_langcube.graf.subgrafs.message_translator_subgraf import MessageTranslatorSubgraf

    message_parts = []
    for comp in components:
        if isinstance(comp, tuple) and len(comp) == 2:
            czech_text, english_text = comp
            if language.startswith("English"):
                message_parts.append(english_text)
            elif language.startswith("Czech"):
                message_parts.append(czech_text)
            else:
                # Fallback translation using English input
                MessageTranslatorSubgraf_instance = MessageTranslatorSubgraf(config=RunnableConfig, subfolder=subfolder)
                translated, tokens = MessageTranslatorSubgraf_instance.run({
                    "translation_input": english_text,
                    "language": language
                })
                logger.info(translated)
                message_parts.append(translated)
        else:
            message_parts.append(str(comp))
    return "\n\n".join(message_parts)
