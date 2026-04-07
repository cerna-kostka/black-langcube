"""
Black LangCube - A framework for building LLM applications with LangGraph.
"""

__version__ = "0.4.0"
__description__ = "A framework for building LLM applications with LangGraph"

# Import core components to make them available from the main package
from .config import ConfigurationError, validate_config
from .graf.graph_base import BaseGraph, GraphState
from .llm_modules.LLMNodes.LLMNode import LLMNode
from .llm_modules.llm_model import LLMProvider, ModelTier
from .helper_modules.get_basegraph_classes import get_basegraph_classes
from .process import run_workflow_by_id, run_complete_pipeline, run_parallel_pipeline

# Expose main components
__all__ = [
    "ConfigurationError",
    "validate_config",
    "BaseGraph",
    "GraphState",
    "LLMNode",
    "LLMProvider",
    "ModelTier",
    "get_basegraph_classes",
    "run_workflow_by_id",
    "run_complete_pipeline",
    "run_parallel_pipeline",
]
