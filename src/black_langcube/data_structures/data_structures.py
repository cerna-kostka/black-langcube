"""
This module defines Pydantic data structures for managing scientific article metadata, search strategies, outlines, chapters, titles & abstracts.

Classes:
    Strategies: Represents up to 10 search strategies as string fields.
    Article: Represents a scientific article with fields for title, findings, topics, knowledge gaps, and points of interest.
    OutlineItem: Represents a single section in an outline, including its hierarchical level, name, and annotation.
    Outline: Represents an outline as a list of up to 8 OutlineItem objects.
    Chapter: Represents a chapter or section in an outline, including its level, name, and a list of paragraphs.
    TitleAbstract: Represents the title and abstract of an article.
"""

from pydantic.v1 import BaseModel, Field
from typing import List


class Strategies(BaseModel):
    strategy1: str = Field(description="strategy blabla")
    strategy2: str = Field(...)


class Article(BaseModel):
    topic: str = Field(description="topic")
    language: str = Field(description="language")
    ... # Add other fields as necessary


class OutlineItem(BaseModel):
    foo: str = Field(description="foo")
    baz: str = Field(description="baz")
    bar: str = Field(description="bar")
    ... # Add other fields as necessary


class Outline(BaseModel):
    items: List[OutlineItem] = Field(
        max_items=8,
        description="List of sections in the outline with a maximum of 8 items"
    )

