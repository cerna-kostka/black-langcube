"""
Black LangCube - A framework for building LLM applications with LangGraph.
"""

__version__ = "0.4.12"
__description__ = "A framework for building LLM applications with LangGraph"

# Import core components to make them available from the main package
from .config import ConfigurationError, validate_config
from .graf.graph_base import BaseGraph, GraphState
from .llm_modules.LLMNodes.LLMNode import LLMNode
from .llm_modules.llm_model import LLMProvider, ModelTier, get_llm_config_summary
from .helper_modules.get_basegraph_classes import get_basegraph_classes
from .process import run_workflow_by_id, run_complete_pipeline, run_parallel_pipeline
from .llm_modules.circuit_breaker_async import (
    CircuitBreakerAsync,
    CircuitBreakerOpenError,
    CircuitBreakerError,
    get_circuit_breaker_async,
    get_all_circuit_breakers,
    reset_all_circuit_breakers,
    reset_circuit_breaker,
)

# Expose main components
__all__ = [
    "ConfigurationError",
    "validate_config",
    "BaseGraph",
    "GraphState",
    "LLMNode",
    "LLMProvider",
    "ModelTier",
    "get_llm_config_summary",
    "get_basegraph_classes",
    "run_workflow_by_id",
    "run_complete_pipeline",
    "run_parallel_pipeline",
    # Circuit breaker
    "CircuitBreakerAsync",
    "CircuitBreakerOpenError",
    "CircuitBreakerError",
    "get_circuit_breaker_async",
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers",
    "reset_circuit_breaker",
]
