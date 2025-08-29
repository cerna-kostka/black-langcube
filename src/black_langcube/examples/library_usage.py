"""
Complete Library Usage Example

This example demonstrates how to use Black LangCube as a library,
including running individual workflows and complete pipelines.
"""

import logging
import sys
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the main processing functions
try:
    from black_langcube.process import (
        run_workflow_by_id,
        run_complete_pipeline,
        cleanup_session
    )
    from black_langcube.graf.graph_base import BaseGraph, GraphState

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the library is installed: pip install -e .")
    sys.exit(1)


def example_single_workflow():
    """Example of running a single workflow."""
    print("=" * 60)
    print("SINGLE WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Define the input
    user_message = "What are the latest developments in artificial intelligence for healthcare?"
    output_folder = Path("./library_output/single_workflow")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {user_message}")
    print(f"Output folder: {output_folder}")
    
    try:
        # Run workflow 1 (question processing and language detection)
        result = run_workflow_by_id(
            user_message=user_message,
            workflow_id=1,
            folder_name=str(output_folder),
            language="English"
        )
        
        print(f"\nWorkflow completed successfully!")
        print(f"Workflow: {result['workflow_name']}")
        print(f"Status: {result['status']}")
        print(f"Result preview: {str(result['result'])[:200]}...")
        
        return result
        
    except Exception as e:
        print(f"Error running workflow: {e}")
        return None


def example_complete_pipeline():
    """Example of running a complete pipeline."""
    print("\n" + "=" * 60)
    print("COMPLETE PIPELINE EXAMPLE")
    print("=" * 60)
    
    # Define the input
    user_message = "How can machine learning improve medical diagnosis accuracy?"
    output_folder = Path("./library_output/complete_pipeline")
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {user_message}")
    print(f"Output folder: {output_folder}")
    
    try:
        # Run a partial pipeline (workflows 1-3)
        results = run_complete_pipeline(
            user_message=user_message,
            folder_name=str(output_folder),
            language="English", 
            start_from=1,
            end_at=3  # Only run the first 3 workflows
        )
        
        print(f"\nPipeline completed!")
        print(f"Total workflows executed: {len(results['pipeline_results'])}")
        print(f"Pipeline status: {results['status']}")
        
        # Show results from each workflow
        for i, workflow_result in enumerate(results['pipeline_results'], 1):
            print(f"\nWorkflow {i}:")
            print(f"  Name: {workflow_result['workflow_name']}")
            print(f"  Status: {workflow_result['status']}")
            print(f"  Result preview: {str(workflow_result['result'])[:100]}...")
        
        return results
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return None


def example_error_handling():
    """Example of error handling and recovery."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING EXAMPLE")
    print("=" * 60)
    
    # Try to run an invalid workflow
    try:
        result = run_workflow_by_id(
            user_message="Test message",
            workflow_id=999,  # Invalid workflow ID
            folder_name="./library_output/error_test"
        )
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")
    
    # Try to run pipeline with invalid range
    try:
        result = run_complete_pipeline(
            user_message="Test message",
            folder_name="./library_output/error_test",
            start_from=5,
            end_at=3  # Invalid range
        )
    except ValueError as e:
        print(f"Caught expected ValueError: {e}")


def example_session_management():
    """Example of session management and cleanup."""
    print("\n" + "=" * 60)
    print("SESSION MANAGEMENT EXAMPLE")
    print("=" * 60)
    
    session_folder = "./library_output/session_test"
    
    try:
        # Run a workflow
        result = run_workflow_by_id(
            user_message="Test session management",
            workflow_id=1,
            folder_name=session_folder
        )
        
        print("Workflow completed, now cleaning up session...")
        
        # Clean up the session
        cleanup_success = cleanup_session(session_folder)
        
        if cleanup_success:
            print("Session cleanup completed successfully")
        else:
            print("Session cleanup failed")
            
    except Exception as e:
        print(f"Error in session management example: {e}")


def demonstrate_library_integration():
    """Show how to integrate the library into a larger application."""
    print("\n" + "=" * 60)
    print("LIBRARY INTEGRATION EXAMPLE")
    print("=" * 60)
    
    class MyApplication:
        """Example application using Black LangCube."""
        
        def __init__(self, base_output_dir: str = "./app_output"):
            self.base_output_dir = Path(base_output_dir)
            self.base_output_dir.mkdir(exist_ok=True)
            self.session_counter = 0
        
        def process_user_query(self, query: str, language: str = "English") -> dict:
            """Process a user query using the library."""
            self.session_counter += 1
            session_folder = self.base_output_dir / f"session_{self.session_counter}"
            
            try:
                # Run the first workflow to process the query
                result = run_workflow_by_id(
                    user_message=query,
                    workflow_id=1,
                    folder_name=str(session_folder),
                    language=language
                )
                
                return {
                    "success": True,
                    "session_id": self.session_counter,
                    "result": result,
                    "message": "Query processed successfully"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "session_id": self.session_counter,
                    "error": str(e),
                    "message": "Failed to process query"
                }
        
        def run_analysis_pipeline(self, query: str) -> dict:
            """Run a complete analysis pipeline."""
            self.session_counter += 1
            session_folder = self.base_output_dir / f"analysis_{self.session_counter}"
            
            try:
                results = run_complete_pipeline(
                    user_message=query,
                    folder_name=str(session_folder),
                    start_from=1,
                    end_at=2  # Run first two workflows
                )
                
                return {
                    "success": True,
                    "session_id": self.session_counter,
                    "pipeline_results": results,
                    "message": "Analysis pipeline completed"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "session_id": self.session_counter, 
                    "error": str(e),
                    "message": "Analysis pipeline failed"
                }
    
    # Demonstrate the application
    app = MyApplication()
    
    # Process a query
    query_result = app.process_user_query(
        "What are the benefits of renewable energy?"
    )
    print(f"Query processing result: {query_result['message']}")
    print(f"Success: {query_result['success']}")
    
    # Run an analysis
    analysis_result = app.run_analysis_pipeline(
        "How does climate change affect agriculture?"
    )
    print(f"Analysis result: {analysis_result['message']}")
    print(f"Success: {analysis_result['success']}")


def main():
    """
    Main function to run all examples.
    """
    print("Black LangCube - Complete Library Usage Examples")
    print("=" * 80)
    
    # Run individual examples
    example_single_workflow()
    example_complete_pipeline()
    example_error_handling()
    example_session_management()
    demonstrate_library_integration()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("Check the ./library_output directory for generated files.")
    print("=" * 80)


if __name__ == "__main__":
    main()
