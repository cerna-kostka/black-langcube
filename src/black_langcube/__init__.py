"""
Black LangCube - A framework for building LLM applications with LangGraph.
"""

__version__ = "0.1.0"
__description__ = "A framework for building LLM applications with LangGraph"

# Import core components to make them available from the main package
from .graf.graph_base import BaseGraph, GraphState
from .llm_modules.LLMNodes.LLMNode import LLMNode
from .helper_modules.get_basegraph_classes import get_basegraph_classes
from .process import run_workflow_by_id, run_complete_pipeline

# Expose main components
__all__ = [
    "BaseGraph",
    "GraphState", 
    "LLMNode",
    "get_basegraph_classes",
    "run_workflow_by_id",
    "run_complete_pipeline",
]