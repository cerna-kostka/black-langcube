"""
Reference dummy implementations for Black LangCube library components.
This file serves as reference documentation for the mock implementations
that were created in the actual graf/ directory files.

The actual dummy implementations are now in:
- ../graf/graf1.py - Graph1 (question processing)
- ../graf/graf2.py - Graph2 (keyword processing)  
- ../graf/graf3.py - Graph3 (strategy generation)
- ../graf/graf4.py - Graph4 (search and analysis)
- ../graf/graf5.py - Graph5 (final processing)

The actual helper functions are used from:
- ../helper_modules/submodules.py - SessionCreator function and end_process function
"""

import logging
from typing import Dict, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)


# Reference implementation patterns used in the actual graf files:

def create_mock_result_base(user_message: str, language: str, workflow_name: str) -> Dict[str, Any]:
    """
    Reference function showing the base structure used in all mock implementations.
    This pattern is used in all graf/*.py files.
    """
    import time
    
    return {
        "input_message": user_message,
        "language": language,
        "workflow": workflow_name,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mock_analysis": f"Analyzed '{user_message[:30]}...' using {workflow_name}",
        "tokens_used": len(user_message.split()) * 10,  # Mock token count
        "confidence": 0.85
    }


def mock_keyword_extraction(user_message: str) -> list:
    """
    Reference function showing keyword extraction pattern used in Graph2.
    """
    words = user_message.lower().split()
    return [word for word in words if len(word) > 4][:5]  # Simple keyword extraction


# Example workflow-specific data patterns:

GRAPH1_MOCK_EXTENSIONS = {
    "detected_language": "English",
    "question_type": "informational",  # or "analytical"
    "entities": ["AI", "healthcare", "technology"],
    "complexity_score": 0.7,
    "processing_nodes": ["language_detector", "entity_extractor", "question_classifier"]
}

GRAPH2_MOCK_EXTENSIONS = {
    "extracted_keywords": [],  # populated by mock_keyword_extraction
    "keyword_weights": {},  # calculated weights
    "semantic_clusters": ["technology", "healthcare", "research"],
    "processing_nodes": ["keyword_extractor", "semantic_analyzer", "weight_calculator"]
}

GRAPH3_MOCK_EXTENSIONS = {
    "research_strategies": [
        "Literature review approach",
        "Case study analysis", 
        "Comparative analysis"
    ],
    "search_terms": ["machine learning", "medical diagnosis", "accuracy"],
    "approach_confidence": 0.82,
    "estimated_research_time": "2-3 hours",
    "processing_nodes": ["strategy_planner", "approach_selector", "term_generator"]
}

GRAPH4_MOCK_EXTENSIONS = {
    "search_results": {
        "total_sources": 25,
        "relevant_sources": 18,
        "academic_papers": 12,
        "web_articles": 6
    },
    "analysis_summary": "Comprehensive analysis of 18 relevant sources on the topic",
    "key_findings": [
        "AI shows 15-20% improvement in diagnostic accuracy",
        "Machine learning models excel in image analysis",
        "Integration challenges remain in clinical practice"
    ],
    "processing_nodes": ["web_searcher", "source_analyzer", "relevance_scorer"]
}

GRAPH5_MOCK_EXTENSIONS = {
    "final_report": {
        "title": "Analysis Report: [user_message]...",
        "sections": ["Introduction", "Methodology", "Findings", "Conclusions"],
        "word_count": 2500,
        "confidence_level": "High"
    },
    "output_formats": ["PDF", "JSON", "HTML"],
    "quality_score": 0.91,
    "processing_nodes": ["report_generator", "formatter", "quality_checker"]
}
