"""
Core Processing Module for Black LangCube

This module contains the main processing functions that orchestrate
different workflow graphs based on input parameters. It serves as
the central entry point for running various LangGraph workflows.
"""

import logging
logger = logging.getLogger(__name__)

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from black_langcube.helper_modules.submodules import SessionCreator, end_process

from black_langcube.graf.graf1 import Graph1
from black_langcube.graf.graf2 import Graph2  
from black_langcube.graf.graf3 import Graph3
from black_langcube.graf.graf4 import Graph4
from black_langcube.graf.graf5 import Graph5


def run_workflow_by_id(
    user_message: str, 
    workflow_id: Union[str, int], 
    folder_name: str, 
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a specific workflow based on the provided workflow ID.
    
    This function orchestrates different LangGraph workflows based on the
    workflow_id parameter and returns structured results.
    
    Args:
        user_message (str): The user input message to process
        workflow_id (str|int): Identifier for which workflow to run (1-5)
        folder_name (str): Directory path for output files
        language (str, optional): Language for processing. Defaults to None.
    
    Returns:
        Dict[str, Any]: Structured result containing workflow output and metadata
        
    Raises:
        ValueError: If workflow_id is not recognized
        RuntimeError: If workflow execution fails
    """
    
    logger.info(f"Starting workflow {workflow_id} with message: {user_message[:100]}...")
    
    # Normalize workflow_id to string for consistent handling
    workflow_id = str(workflow_id)
    
    # Create session and prepare folder structure
    session_result = SessionCreator()
    # Use the provided folder_name instead of the auto-generated one
    folder_path = Path(folder_name)
    folder_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if workflow_id == "1":
            return _run_graph1(user_message, folder_name, language)
        elif workflow_id == "2":
            return _run_graph2(user_message, folder_name, language)
        elif workflow_id == "3":
            return _run_graph3(user_message, folder_name, language)
        elif workflow_id == "4":
            return _run_graph4(user_message, folder_name, language)
        elif workflow_id == "5":
            return _run_graph5(user_message, folder_name, language)
        else:
            raise ValueError(f"Unknown workflow ID: {workflow_id}")
            
    except Exception as e:
        logger.error(f"Error in workflow {workflow_id}: {str(e)}")
        raise RuntimeError(f"Workflow {workflow_id} execution failed: {str(e)}") from e


def _run_graph1(user_message: str, folder_name: str, language: Optional[str]) -> Dict[str, Any]:
    """Run Graph1 workflow for initial question processing and language detection."""
    logger.info("Running Graph1 - Question processing and language detection")
    
    graph1 = Graph1(user_message, folder_name, language)
    result = graph1.run()
    
    return {
        "workflow_id": "1",
        "workflow_name": "question_processing", 
        "result": result,
        "status": "completed",
        "output_folder": folder_name
    }


def _run_graph2(user_message: str, folder_name: str, language: Optional[str]) -> Dict[str, Any]:
    """Run Graph2 workflow for keyword processing."""
    logger.info("Running Graph2 - Keyword processing")
    
    graph2 = Graph2(user_message, folder_name, language)
    result = graph2.run()
    
    return {
        "workflow_id": "2", 
        "workflow_name": "keyword_processing",
        "result": result,
        "status": "completed",
        "output_folder": folder_name
    }


def _run_graph3(user_message: str, folder_name: str, language: Optional[str]) -> Dict[str, Any]:
    """Run Graph3 workflow for strategy generation."""
    logger.info("Running Graph3 - Strategy generation")
    
    graph3 = Graph3(user_message, folder_name, language)
    result = graph3.run()
    
    return {
        "workflow_id": "3",
        "workflow_name": "strategy_generation", 
        "result": result,
        "status": "completed",
        "output_folder": folder_name
    }


def _run_graph4(user_message: str, folder_name: str, language: Optional[str]) -> Dict[str, Any]:
    """Run Graph4 workflow for search and analysis."""
    logger.info("Running Graph4 - Search and analysis")
    
    graph4 = Graph4(user_message, folder_name, language)
    result = graph4.run()
    
    return {
        "workflow_id": "4",
        "workflow_name": "search_and_analysis",
        "result": result,
        "status": "completed", 
        "output_folder": folder_name
    }


def _run_graph5(user_message: str, folder_name: str, language: Optional[str]) -> Dict[str, Any]:
    """Run Graph5 workflow for final processing and output generation."""
    logger.info("Running Graph5 - Final processing and output generation")
    
    graph5 = Graph5(user_message, folder_name, language)
    result = graph5.run()
    
    return {
        "workflow_id": "5",
        "workflow_name": "final_processing",
        "result": result,
        "status": "completed",
        "output_folder": folder_name
    }


def run_complete_pipeline(
    user_message: str,
    folder_name: str, 
    language: Optional[str] = None,
    start_from: int = 1,
    end_at: int = 5
) -> Dict[str, Any]:
    """
    Run a complete pipeline of workflows from start_from to end_at.
    
    Args:
        user_message (str): Input message to process
        folder_name (str): Output directory
        language (str, optional): Processing language
        start_from (int): Starting workflow number (1-5)
        end_at (int): Ending workflow number (1-5)
        
    Returns:
        Dict[str, Any]: Results from all executed workflows
    """
    logger.info(f"Running complete pipeline from workflow {start_from} to {end_at}")
    
    if start_from < 1 or end_at > 5 or start_from > end_at:
        raise ValueError("Invalid workflow range. Must be between 1-5 and start_from <= end_at")
    
    results = {
        "pipeline_results": [],
        "total_workflows": end_at - start_from + 1,
        "status": "completed",
        "folder_name": folder_name
    }
    
    for workflow_id in range(start_from, end_at + 1):
        try:
            result = run_workflow_by_id(user_message, workflow_id, folder_name, language)
            results["pipeline_results"].append(result)
            logger.info(f"Completed workflow {workflow_id}")
        except Exception as e:
            logger.error(f"Pipeline failed at workflow {workflow_id}: {str(e)}")
            results["status"] = "failed"
            results["failed_at_workflow"] = workflow_id
            results["error"] = str(e)
            break
    
    return results


def cleanup_session(folder_name: str) -> bool:
    """
    Clean up session files and temporary data.
    
    Args:
        folder_name (str): Session folder to clean up
        
    Returns:
        bool: True if cleanup was successful
    """
    try:
        # The end_process function expects user_message, folder_name, and language
        # For cleanup purposes, we can use placeholder values
        end_process("cleanup", folder_name, "English")
        logger.info(f"Session cleanup completed for {folder_name}")
        return True
    except Exception as e:
        logger.error(f"Error during session cleanup: {str(e)}")
        return False
