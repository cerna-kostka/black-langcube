"""
Custom LLM Node Example

This example demonstrates how to create custom LLM nodes for specific tasks
and integrate them into workflows.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Set up logging with more detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure output goes to stdout
    ],
    force=True  # Force reconfiguration if already configured
)

# Also set up root logger to catch all messages
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

from dotenv import load_dotenv
load_dotenv()

# Import the core components
from black_langcube.graf.graph_base import BaseGraph, GraphState
from black_langcube.llm_modules.LLMNodes.LLMNode import LLMNode

from langgraph.graph import START, END
from langchain_core.messages import HumanMessage


class TextAnalysisState(GraphState):
    """State for text analysis workflow."""
    input_text: str
    analysis_result: Dict[str, Any]
    summary: str
    sentiment: str
    key_points: list


class TextAnalyzerNode(LLMNode):
    """
    Custom LLM node for analyzing text sentiment and extracting key information.
    """
    
    def generate_messages(self):
        """Generate messages for the LLM."""
        text = self.state.get("input_text", "")
        
        system_prompt = """You are an expert text analyzer. Analyze the given text and provide:
1. Overall sentiment (positive, negative, neutral)
2. Key themes and topics
3. Important insights or conclusions
4. A brief summary

Format your response as a structured analysis."""
        
        return [
            ("system", system_prompt),
            ("human", f"Please analyze this text:\n\n{text}")
        ]
    
    def execute(self, extra_input=None):
        """Execute the text analysis."""
        self.logger.info("Analyzing text with LLM...")
        
        # Run the LLM chain
        result, tokens = self.run_chain(extra_input or {})
        
        # Parse the result (in a real implementation, you might use structured output)
        analysis = {
            "raw_analysis": result,
            "tokens_used": tokens,
            "analysis_type": "comprehensive"
        }
        
        # Update state
        self.state["analysis_result"] = analysis
        
        self.logger.info("Text analysis completed")
        return {"analysis_result": analysis}


class SummaryExtractorNode(LLMNode):
    """
    Custom node to extract a concise summary from the analysis.
    """
    
    def generate_messages(self):
        """Generate messages for summary extraction."""
        analysis = self.state.get("analysis_result", {})
        raw_analysis = analysis.get("raw_analysis", "")
        
        system_prompt = """Extract a concise 2-3 sentence summary from the following analysis. 
Focus on the main points and conclusions."""
        
        return [
            ("system", system_prompt),
            ("human", f"Analysis to summarize:\n\n{raw_analysis}")
        ]
    
    def execute(self, extra_input=None):
        """Execute summary extraction."""
        self.logger.info("Extracting summary...")
        
        result, tokens = self.run_chain(extra_input or {})
        
        # Update state
        self.state["summary"] = result
        
        self.logger.info("Summary extraction completed")
        return {"summary": result}


class SentimentExtractorNode(LLMNode):
    """
    Custom node to extract sentiment classification.
    """
    
    def generate_messages(self):
        """Generate messages for sentiment extraction."""
        text = self.state.get("input_text", "")
        
        system_prompt = """Classify the sentiment of the given text. 
Respond with only one word: 'positive', 'negative', or 'neutral'."""
        
        return [
            ("system", system_prompt),
            ("human", text)
        ]
    
    def execute(self, extra_input=None):
        """Execute sentiment classification."""
        self.logger.info("Classifying sentiment...")
        
        result, tokens = self.run_chain(extra_input or {})
        
        # Clean up the result
        sentiment = result.strip().lower()
        if sentiment not in ['positive', 'negative', 'neutral']:
            sentiment = 'neutral'  # Default fallback
        
        self.state["sentiment"] = sentiment
        
        self.logger.info(f"Sentiment classified as: {sentiment}")
        return {"sentiment": sentiment}


class TextAnalysisWorkflow(BaseGraph):
    """
    A workflow that demonstrates custom LLM nodes for text analysis.
    """
    
    def __init__(self, input_text: str, folder_name: str):
        super().__init__(TextAnalysisState, input_text, folder_name, "English")
        self.state = TextAnalysisState()
        self.state["input_text"] = input_text
        self.state["messages"] = [HumanMessage(input_text)]
        self.build_graph()
    
    def _log_and_execute(self, node_name, func, state):
        """Log entry and exit of a node and execute it."""
        logger = logging.getLogger(__name__)
        logger.info(f"Entering node: {node_name}")
        
        # The `execute` methods on the nodes don't take state, but `finalize_analysis` does.
        # A simpler way is to check the function name.
        if "finalize" in node_name:
            result = func(state)
        else:
            # The node's execute method updates the state internally.
            result = func()
            
        logger.info(f"Exiting node: {node_name}")
        return result

    def build_graph(self):
        """Build the text analysis workflow."""
        logger = logging.getLogger(__name__)
        logger.info("Building text analysis workflow graph...")
        
        # Create custom node instances
        analyzer_node = TextAnalyzerNode(self.state, {})
        summary_node = SummaryExtractorNode(self.state, {})
        sentiment_node = SentimentExtractorNode(self.state, {})
        
        logger.info("Created custom LLM node instances")
        
        def finalize_analysis(state):
            """Finalize the analysis by combining results."""
            logger = logging.getLogger(__name__)
            
            result = {
                "input_text": state.get("input_text", ""),
                "sentiment": state.get("sentiment", "unknown"),
                "summary": state.get("summary", ""),
                "full_analysis": state.get("analysis_result", {}),
                "workflow_complete": True
            }
            
            logger.info("Finalizing analysis results...")
            logger.info(f"Detected sentiment: {result['sentiment']}")
            logger.info(f"Generated summary: {result['summary'][:50]}...")
            
            print("\n" + "="*50)
            print("ANALYSIS COMPLETE")
            print("="*50)
            print(f"Input Text: {result['input_text'][:100]}...")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Summary: {result['summary']}")
            print("="*50)
            
            return result
        
        # Add nodes to the graph
        self.add_node("analyze_text", lambda state: self._log_and_execute("analyze_text", analyzer_node.execute, state))
        self.add_node("extract_summary", lambda state: self._log_and_execute("extract_summary", summary_node.execute, state))
        self.add_node("classify_sentiment", lambda state: self._log_and_execute("classify_sentiment", sentiment_node.execute, state))
        self.add_node("finalize", lambda state: self._log_and_execute("finalize", finalize_analysis, state))
        
        logger.info("Added nodes to workflow graph")
        
        # Define the workflow
        self.add_edge(START, "analyze_text")
        self.add_edge("analyze_text", "extract_summary")
        self.add_edge("extract_summary", "classify_sentiment")
        self.add_edge("classify_sentiment", "finalize")
        self.add_edge("finalize", END)
        
        logger.info("Workflow graph construction completed")
    
    @property
    def workflow_name(self):
        return "text_analysis_workflow"

    def run(self):
        """Run the text analysis workflow."""
        self.intro_info_check()
        initial_state = dict(self.state)
        
        # Consume the stream to execute the graph
        for _ in self.graph_streaming(initial_state):
            pass

        # The final state is available in self.state after execution
        return self.state


def main():
    """
    Demonstrate custom LLM nodes in a text analysis workflow.
    """
    # Set up logger for main function
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("Black LangCube - Custom LLM Node Example")
    print("=" * 60)
    
    logger.info("Starting text analysis workflow demonstration")
    
    # Sample texts for analysis
    sample_texts = [
        """
        The new artificial intelligence system has shown remarkable improvements in 
        processing natural language. Users report high satisfaction levels and 
        significant time savings in their daily tasks. The technology represents 
        a major breakthrough in making AI more accessible to everyone.
        """,
        """
        The project has faced numerous setbacks and budget overruns. Team morale 
        is low and several key developers have left the company. The deadline 
        seems impossible to meet given the current circumstances and lack of resources.
        """,
        """
        The quarterly earnings report shows steady growth in revenue but also 
        indicates some challenges in the competitive market. The company maintains 
        a stable position while exploring new opportunities for expansion.
        """
    ]
    
    # Create output directory
    output_dir = Path("./analysis_output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    # Process each text
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'='*20} ANALYSIS {i} {'='*20}")
        logger.info(f"Starting analysis {i} of {len(sample_texts)}")
        
        workflow = TextAnalysisWorkflow(
            input_text=text.strip(),
            folder_name=str(output_dir / f"analysis_{i}")
        )
        
        try:
            logger.info(f"Running workflow {i}...")
            result = workflow.run()
            print(f"Workflow {i} completed successfully!")
            logger.info(f"Workflow {i} completed successfully!")
            
        except Exception as e:
            print(f"Error in analysis {i}: {e}")
            logger.error(f"Error in analysis {i}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("All text analysis workflows completed!")
    print("\n" + "=" * 60)
    print("All text analysis workflows completed!")


if __name__ == "__main__":
    main()
