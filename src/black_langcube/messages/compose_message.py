import logging
from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


async def compose_message(language, components, subfolder=None):
    """
    Compose a message for a specific language given a list of components.

    Each component can either be:
      - A tuple (english_text, czech_text) representing the text in English and Czech.
        For languages other than English or Czech, the English version is used as the base for translation.
      - A string (or any object that can be cast to a string) that is included as is.

    The components are joined by double newlines.
    """

    from black_langcube.graf.subgrafs.message_translator_subgraf import (
        MessageTranslatorSubgraf,
    )

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
                MessageTranslatorSubgraf_instance = MessageTranslatorSubgraf(
                    config=RunnableConfig, subfolder=subfolder
                )
                result = await MessageTranslatorSubgraf_instance.run(
                    {"translation_input": english_text, "language": language}
                )
                if isinstance(result, tuple) and len(result) >= 1:
                    translated = result[0]
                else:
                    translated = str(result)
                logger.debug(translated)
                message_parts.append(translated)
        else:
            if isinstance(comp, dict) and "error" in comp:
                logger.warning(f"Error in message component: {comp['error']}")
                message_parts.append(f"[Error: {comp['error']}]")
            else:
                message_parts.append(str(comp))
    return "\n\n".join(message_parts)
