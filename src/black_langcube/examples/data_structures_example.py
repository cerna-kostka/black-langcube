"""
Data Structures Example

This example demonstrates how to use and extend the built-in Pydantic data structures
for scientific article processing and other structured data tasks.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Optional
from black_langcube.data_structures.data_structures import (
    Strategies,
    Article,
    Outline,
    OutlineItem
)

from pydantic.v1 import BaseModel, Field, ValidationError


def demonstrate_strategies():
    """Demonstrate the Strategies data structure."""
    print("=" * 50)
    print("STRATEGIES DATA STRUCTURE")
    print("=" * 50)
    
    # Create a strategies instance
    strategies = Strategies(
        strategy1="Search for peer-reviewed articles on machine learning",
        strategy2="Focus on recent publications from 2020-2024"
    )
    
    print("Created strategies:")
    print(f"  Strategy 1: {strategies.strategy1}")
    print(f"  Strategy 2: {strategies.strategy2}")
    
    # Convert to dict and JSON
    strategies_dict = strategies.dict()
    strategies_json = strategies.json(indent=2)
    
    print(f"\nAs dictionary: {strategies_dict}")
    print(f"\nAs JSON:\n{strategies_json}")
    
    return strategies


def demonstrate_article():
    """Demonstrate the Article data structure."""
    print("\n" + "=" * 50)
    print("ARTICLE DATA STRUCTURE")
    print("=" * 50)
    
    # Create an article instance
    article = Article(
        topic="Artificial Intelligence in Healthcare",
        language="English"
    )
    
    print("Created article:")
    print(f"  Topic: {article.topic}")
    print(f"  Language: {article.language}")
    
    # Demonstrate validation
    try:
        invalid_article = Article(
            topic="",  # Empty topic should be caught by validation
            language="English"
        )
    except ValidationError as e:
        print(f"\nValidation caught empty topic: {e}")
    
    return article


def demonstrate_outline():
    """Demonstrate the Outline and OutlineItem data structures."""
    print("\n" + "=" * 50)
    print("OUTLINE DATA STRUCTURES")
    print("=" * 50)
    
    # Create outline items
    items = [
        OutlineItem(
            foo="Introduction",
            baz="1",
            bar="Overview of the research problem"
        ),
        OutlineItem(
            foo="Literature Review",
            baz="2",
            bar="Analysis of existing work in the field"
        ),
        OutlineItem(
            foo="Methodology",
            baz="3",
            bar="Research approach and methods used"
        ),
        OutlineItem(
            foo="Results",
            baz="4",
            bar="Key findings and data analysis"
        ),
        OutlineItem(
            foo="Discussion",
            baz="5",
            bar="Interpretation of results and implications"
        ),
        OutlineItem(
            foo="Conclusion",
            baz="6",
            bar="Summary and future research directions"
        )
    ]
    
    # Create an outline
    outline = Outline(items=items)
    
    print(f"Created outline with {len(outline.items)} sections:")
    for i, item in enumerate(outline.items, 1):
        print(f"  {i}. {item.foo} (Level {item.baz}): {item.bar}")
    
    # Test max items validation
    try:
        too_many_items = [OutlineItem(foo=f"Section {i}", baz=str(i), bar=f"Description {i}") 
                         for i in range(1, 11)]  # 10 items, max is 8 so it will fail
        print("\nAttempting to create outline with too many items...")
        # This should raise a ValidationError
        invalid_outline = Outline(items=too_many_items)
    except ValidationError as e:
        print(f"\nValidation caught too many items: {e}")
    
    return outline


def create_custom_data_structure():
    """Demonstrate creating custom data structures."""
    print("\n" + "=" * 50)
    print("CUSTOM DATA STRUCTURES")
    print("=" * 50)
    
    class ResearchPaper(BaseModel):
        """Custom data structure for research papers."""
        title: str = Field(description="Title of the paper")
        authors: List[str] = Field(description="List of authors")
        abstract: str = Field(description="Paper abstract")
        keywords: List[str] = Field(description="Research keywords")
        year: int = Field(description="Publication year", ge=1900, le=2030)
        journal: Optional[str] = Field(None, description="Journal name")
        doi: Optional[str] = Field(None, description="Digital Object Identifier")
        
        class Config:
            schema_extra = {
                "example": {
                    "title": "Machine Learning in Medical Diagnosis",
                    "authors": ["Dr. Smith", "Dr. Johnson"],
                    "abstract": "This paper explores...",
                    "keywords": ["machine learning", "medical diagnosis", "AI"],
                    "year": 2024,
                    "journal": "Journal of Medical AI",
                    "doi": "10.1000/example"
                }
            }
    
    class ResearchDatabase(BaseModel):
        """Custom data structure for a collection of papers."""
        papers: List[ResearchPaper] = Field(description="List of research papers")
        total_count: int = Field(description="Total number of papers")
        search_query: str = Field(description="Original search query")
        
        def add_paper(self, paper: ResearchPaper):
            """Add a paper to the database."""
            self.papers.append(paper)
            self.total_count = len(self.papers)
        
        def get_papers_by_year(self, year: int) -> List[ResearchPaper]:
            """Get papers published in a specific year."""
            return [paper for paper in self.papers if paper.year == year]
    
    # Create sample papers
    paper1 = ResearchPaper(
        title="Deep Learning for Medical Image Analysis",
        authors=["Dr. Alice Cooper", "Dr. Bob Smith"],
        abstract="This study investigates the application of deep learning techniques...",
        keywords=["deep learning", "medical imaging", "neural networks"],
        year=2023,
        journal="Medical AI Quarterly",
        doi="10.1000/example1"
    )
    
    paper2 = ResearchPaper(
        title="Natural Language Processing in Clinical Notes",
        authors=["Dr. Carol Davis", "Dr. David Wilson"],
        abstract="We present a comprehensive analysis of NLP methods for clinical text...",
        keywords=["NLP", "clinical notes", "text mining"],
        year=2024,
        journal="Clinical Informatics Review",
        doi="10.1000/example2"
    )
    
    # Create a research database
    database = ResearchDatabase(
        papers=[paper1, paper2],
        total_count=2,
        search_query="AI in healthcare"
    )
    
    print(f"Created research database with {database.total_count} papers:")
    for i, paper in enumerate(database.papers, 1):
        print(f"\n  Paper {i}:")
        print(f"    Title: {paper.title}")
        print(f"    Authors: {', '.join(paper.authors)}")
        print(f"    Year: {paper.year}")
        print(f"    Keywords: {', '.join(paper.keywords)}")
    
    # Demonstrate custom methods
    papers_2024 = database.get_papers_by_year(2024)
    print(f"\nPapers from 2024: {len(papers_2024)}")
    
    return database


def save_and_load_example(strategies, article, outline, database):
    """Demonstrate saving and loading data structures."""
    print("\n" + "=" * 50)
    print("SAVING AND LOADING DATA")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("./data_structures_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save each structure
    with open(output_dir / "strategies.json", "w") as f:
        json.dump(strategies.dict(), f, indent=2)
    
    with open(output_dir / "article.json", "w") as f:
        json.dump(article.dict(), f, indent=2)
    
    with open(output_dir / "outline.json", "w") as f:
        json.dump(outline.dict(), f, indent=2)
    
    with open(output_dir / "database.json", "w") as f:
        json.dump(database.dict(), f, indent=2)
    
    print(f"Saved all data structures to {output_dir}")
    
    # Load and verify
    with open(output_dir / "strategies.json", "r") as f:
        loaded_strategies_data = json.load(f)
        loaded_strategies = Strategies(**loaded_strategies_data)
    
    print(f"\nLoaded strategies: {loaded_strategies.strategy1}")
    
    # List all saved files
    saved_files = list(output_dir.glob("*.json"))
    print(f"Saved files: {[f.name for f in saved_files]}")


def main():
    """
    Main function to demonstrate all data structure features.
    """
    print("Black LangCube - Data Structures Example")
    
    # Demonstrate built-in structures
    strategies = demonstrate_strategies()
    article = demonstrate_article()
    outline = demonstrate_outline()
    
    # Demonstrate custom structures
    database = create_custom_data_structure()
    
    # Demonstrate persistence
    save_and_load_example(strategies, article, outline, database)
    
    print("\n" + "=" * 50)
    print("Data structures example completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
