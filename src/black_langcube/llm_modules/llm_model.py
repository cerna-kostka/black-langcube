"""
This module initializes and configures language model instances using the ChatOpenAI class from the langchain_openai package.
It loads environment variables from a local .env file to securely manage credentials and settings.
Two model names are specified: one for a lightweight model and one for a high-performance model, with recommendations to periodically check for updates in OpenAI's available models.
The module provides instantiated model objects for use throughout the application, including default and specialized aliases for various tasks.

    - langchain_openai.ChatOpenAI: Interface for OpenAI's chat-based language models.
    - dotenv.load_dotenv, dotenv.find_dotenv: Utilities for loading environment variables from a .env file.

    - model_name_low (str): Name of the lightweight language model (default: "gpt-4o-mini").
    - model_name_high (str): Name of the high-performance language model (default: "gpt-4.1").
    - llm_low (ChatOpenAI): Instance of ChatOpenAI initialized with the lightweight model.
    - llm_high (ChatOpenAI): Instance of ChatOpenAI initialized with the high-performance model.
    - default_llm (ChatOpenAI): Alias for the default language model (llm_low), used in various nodes.
    - llm_analyst, llm_outline, llm_text, llm_check_title, llm_title_abstract (ChatOpenAI): Aliases for the high-performance model, used for specialized tasks.
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

model_name_low = "gpt-4o-mini" # this by default should be "gpt-4o-mini-2024-07-18" - changes in openAI available model policies should be checked periodically 
model_name_high = "gpt-4.1" # this by default should be "gpt-4o-2024-08-06" - changes in openAI available model policies should be checked periodically

# Model
llm_low = ChatOpenAI(model=model_name_low)
llm_high = ChatOpenAI(model=model_name_high)

# aliases for the models:

default_llm = llm_low
# the default is used at:
# llm_IsLanguageNode
# llm_TranslateQuestionNode
# ...

llm_analyst = llm_high
llm_outline = llm_high
# ...