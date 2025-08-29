from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from black_langcube.llm_modules.llm_model import llm_low

def CheckTitleRelevance(title, topic):
    """
    Check if the given article title is relevant to the specified topic.

    Args:
        title (str): The title of the article.
        topic (str): The topic to check relevance against.

    Returns:
        str: 'Yes' if the title is relevant to the topic, otherwise 'No'.
    """
    messages = [
        ("human", "Is article called '{title}' relevant for topic '{topic}' ? Answer Yes/No only:"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | llm_low | StrOutputParser()

    with get_openai_callback() as cb:
        result = chain.invoke({"title": title, "topic": topic})

    tokens = {
        "tokens_in": cb.prompt_tokens,
        "tokens_out": cb.completion_tokens,
        "tokens_price": cb.total_cost,
    }

    return result, tokens
