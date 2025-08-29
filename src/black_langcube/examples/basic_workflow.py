"""
Basic Usage Example for Black LangCube

This example demonstrates how to create a simple workflow using the BaseGraph class.
"""

import os
import sys
import logging
from pathlib import Path
from typing_extensions import TypedDict

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# You'll need to set up your environment variables
# Create a .env file with your OpenAI API key
from dotenv import load_dotenv
load_dotenv()

# Import the core components
try:
    from black_langcube.graf.graph_base import BaseGraph, GraphState
    from langgraph.graph import START, END
    from langchain_core.messages import HumanMessage
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you've installed the package: pip install -e .")
    exit(1)


# Define a custom state for our workflow
class SimpleGraphState(GraphState):
    user_input: str
    processed_output: str
    step_count: int
    messages: list


class SimpleWorkflow(BaseGraph):
    """
    A simple workflow that demonstrates basic graph construction.
    """
    
    def __init__(self, user_message: str, folder_name: str, language: str = "English"):
        super().__init__(SimpleGraphState, user_message, folder_name, language)
        self.state = {
            "user_input": user_message,
            "processed_output": "",
            "step_count": 0,
            "messages": [HumanMessage(user_message)]
        }
        self.build_graph()
    
    def build_graph(self):
        """Build the workflow graph with nodes and edges."""
        
        def step_1_process_input(state):
            """First processing step."""
            print(f"Step 1: Processing input: {state['user_input']}")
            state["step_count"] += 1
            return {
                "processed_output": f"Processed: {state['user_input']}",
                "step_count": state["step_count"]
            }
        
        def step_2_enhance_output(state):
            """Second processing step."""
            print(f"Step 2: Enhancing output: {state['processed_output']}")
            state["step_count"] += 1
            enhanced = f"Enhanced [{state['processed_output']}] with additional context"
            return {
                "processed_output": enhanced,
                "step_count": state["step_count"]
            }
        
        def step_3_finalize(state):
            """Final processing step."""
            print(f"Step 3: Finalizing: {state['processed_output']}")
            state["step_count"] += 1
            final = f"Final result: {state['processed_output']} (completed in {state['step_count']} steps)"
            return {
                "processed_output": final,
                "step_count": state["step_count"]
            }
        
        # Add nodes to the graph
        self.add_node("process_input", step_1_process_input)
        self.add_node("enhance_output", step_2_enhance_output)
        self.add_node("finalize", step_3_finalize)
        
        # Define the flow with edges
        self.add_edge(START, "process_input")
        self.add_edge("process_input", "enhance_output")
        self.add_edge("enhance_output", "finalize")
        self.add_edge("finalize", END)
    
    @property
    def workflow_name(self):
        return "simple_workflow"
    
    def run(self):
        """
        Run the workflow with the initial state.
        """
        # Set up the initial state
        initial_state = {
            "user_input": self.state["user_input"],
            "processed_output": "",
            "step_count": self.state["step_count"],
            "messages": self.state["messages"]
        }
        
        # Execute the graph
        final_state = None
        for event in self.graph.stream(initial_state):
            # Update the state with the latest event
            for node_name, node_output in event.items():
                # Update our instance state with the node output
                if isinstance(node_output, dict):
                    self.state.update(node_output)
                    final_state = node_output
        
        # Return the final processed output
        return self.state.get("processed_output", "No output generated")


def main():
    """
    Main function to demonstrate the workflow.
    """
    print("=" * 60)
    print("Black LangCube - Basic Workflow Example")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create and run the workflow
    user_message = "Hello, Black LangCube! This is a test message."
    workflow = SimpleWorkflow(
        user_message=user_message,
        folder_name=str(output_dir),
        language="English"
    )
    
    print(f"\nStarting workflow with message: '{user_message}'")
    print("-" * 60)
    
    try:
        # Run the workflow
        result = workflow.run()
        
        print("-" * 60)
        print("Workflow completed successfully!")
        print(f"Result: {result}")
        
        # Show the final state
        final_state = workflow.state
        print(f"\nFinal state:")
        print(f"  - Input: {final_state.get('user_input', 'N/A')}")
        print(f"  - Output: {final_state.get('processed_output', 'N/A')}")
        print(f"  - Steps: {final_state.get('step_count', 0)}")
        
    except Exception as e:
        print(f"Error running workflow: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
