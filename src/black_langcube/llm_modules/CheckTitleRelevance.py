import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from black_langcube.llm_modules.llm_model import default_llm

logger = logging.getLogger(__name__)


def CheckTitleRelevance(title, topic, llm=None):
    """
    Check if the given article title is relevant to the specified topic.

    Args:
        title (str): The title of the article.
        topic (str): The topic to check relevance against.
        llm (BaseChatModel | None): Language model to use. Defaults to ``default_llm()``.

    Returns:
        str: 'Yes' if the title is relevant to the topic, otherwise 'No'.
    """
    model = llm if llm is not None else default_llm()

    messages = [
        (
            "human",
            "Is article called '{title}' relevant for topic '{topic}' ? Answer Yes/No only:",
        ),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | model | StrOutputParser()

    with get_openai_callback() as cb:
        result = chain.invoke({"title": title, "topic": topic})

    tokens = {
        "tokens_in": cb.prompt_tokens,
        "tokens_out": cb.completion_tokens,
        "tokens_price": cb.total_cost,
    }

    return result, tokens
