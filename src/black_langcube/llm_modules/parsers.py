"""
This module sets up output parsers for various data structures using LangChain's output parsers.

Imports:
    - StrOutputParser, JsonOutputParser: Output parsers from langchain_core.
    - Strategies, Article, Outline, Chapter, TitleAbstract: Pydantic data structures from the local data_structures module.

Parsers:
    - parser_strategies: Parses outputs into the Strategies data structure.
    - parser_analyst: Parses outputs into the Article data structure.
    - parser_outline: Parses outputs into the Outline data structure.
    - parser_chapter: Parses outputs into the Chapter data structure.
    - parser_title_abstract: Parses outputs into the TitleAbstract data structure.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser

from black_langcube.data_structures.data_structures import (
    Strategies,
    Article,
    Outline,
)

# Set up parsers
parser_strategies = JsonOutputParser(pydantic_object=Strategies)
parser_analyst = JsonOutputParser(pydantic_object=Article)
parser_outline = JsonOutputParser(pydantic_object=Outline)
