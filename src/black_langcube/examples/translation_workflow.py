"""
Translation Workflow Example

This example demonstrates how to use the translation subgraphs 
and LLM nodes for building a translation pipeline.
"""

import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment setup (you'll need your OpenAI API key)
from dotenv import load_dotenv
load_dotenv()

# Core imports
from black_langcube.graf.subgrafs.translator_en_subgraf import TranslatorEnSubgraf
from black_langcube.graf.subgrafs.translator_usr_subgraf import TranslatorUsrSubgraf
from black_langcube.graf.graph_base import BaseGraph, GraphState
from black_langcube.helper_modules.get_result_from_graph_outputs import get_result_from_graph_outputs

from langgraph.graph import START, END
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from typing import Literal


class TranslationWorkflowState(GraphState):
    """State for the translation workflow."""
    source_text: str
    source_language: str
    target_language: str
    translated_text: str
    translation_tokens: dict


# Custom message functions that use the correct parsing approach
def message_translator_en_fixed(language, subfolder, output_filename):
    """
    Returns a message for translator_en workflow using correct key extraction.
    """
    translation_out = get_result_from_graph_outputs(
        "translator_en",
        "",
        "translation_output",
        "",
        subfolder,
        output_filename
    )
    tokens_out = get_result_from_graph_outputs(
        "translator_en",
        "",
        "translation_tokens",
        "",
        subfolder,
        output_filename
    )
    message = translation_out, tokens_out
    return message


def message_translator_usr_fixed(language, subfolder, output_filename):
    """
    Returns a message for translator_usr workflow using correct key extraction.
    """
    translation_out = get_result_from_graph_outputs(
        "translator_usr",
        "",
        "translation_output",
        "",
        subfolder,
        output_filename
    )
    tokens_out = get_result_from_graph_outputs(
        "translator_usr",
        "",
        "translation_tokens",
        "",
        subfolder,
        output_filename
    )
    message = translation_out, tokens_out
    return message


class TranslationWorkflow(BaseGraph):
    """
    A workflow that demonstrates translation capabilities using subgraphs.
    """
    
    def __init__(self, source_text: str, source_language: str, target_language: str, folder_name: str):
        super().__init__(TranslationWorkflowState, source_text, folder_name, target_language)
        self.state = TranslationWorkflowState()
        self.state["source_text"] = source_text
        self.state["source_language"] = source_language
        self.state["target_language"] = target_language
        self.state["messages"] = [HumanMessage(source_text)]
        self.build_graph()
    
    def build_graph(self):
        """Build the translation workflow."""
        
        def detect_translation_direction(state) -> Literal["translate_to_english", "translate_to_user_language"]:
            """Determine which translation subgraph to use."""
            if state["target_language"].lower().startswith("english"):
                return "translate_to_english"
            else:
                return "translate_to_user_language"
        
        def translate_to_english_node(state):
            """Translate text to English using the English translator subgraph."""
            logger.info("Translating to English...")
            
            config = RunnableConfig()
            subfolder = Path(self.folder_name) / "translation_en"
            
            # Create custom subgraph that uses the fixed message function
            translator = CustomTranslatorEnSubgraf(config, subfolder=str(subfolder))
            
            result = translator.run(extra_input={
                "translation_input": state["source_text"],
                "language": state["source_language"]
            })
            
            # Result is typically a tuple of (translated_text, tokens)
            if isinstance(result, tuple) and len(result) == 2:
                translated_text, tokens = result
            else:
                translated_text = str(result)
                tokens = {}
            
            # Update both the parameter state and self.state
            state["translated_text"] = translated_text
            state["translation_tokens"] = tokens
            self.state["translated_text"] = translated_text
            self.state["translation_tokens"] = tokens
            
            logger.info(f"Translation completed: {translated_text}")
            
            return {
                "translated_text": translated_text,
                "translation_tokens": tokens
            }
        
        def translate_to_user_language_node(state):
            """Translate text to user language using the user translator subgraph."""
            logger.info(f"Translating to {state['target_language']}...")
            
            config = RunnableConfig()
            subfolder = Path(self.folder_name) / "translation_usr"
            
            # Create custom subgraph that uses the fixed message function
            translator = CustomTranslatorUsrSubgraf(config, subfolder=str(subfolder))
            
            result = translator.run(extra_input={
                "translation_input": state["source_text"],
                "language": state["target_language"]
            })
            
            # Result is typically a tuple of (translated_text, tokens)
            if isinstance(result, tuple) and len(result) == 2:
                translated_text, tokens = result
            else:
                translated_text = str(result)
                tokens = {}
            
            # Update both the parameter state and self.state
            state["translated_text"] = translated_text
            state["translation_tokens"] = tokens
            self.state["translated_text"] = translated_text
            self.state["translation_tokens"] = tokens
            
            logger.info(f"Translation completed: {translated_text}")
            
            return {
                "translated_text": translated_text,
                "translation_tokens": tokens
            }
        
        # Add nodes
        self.add_node("translate_to_english", translate_to_english_node)
        self.add_node("translate_to_user_language", translate_to_user_language_node)
        
        # Add conditional edges based on target language
        # Using self.workflow.add_conditional_edges directly like in graf1.py
        self.workflow.add_conditional_edges(START, detect_translation_direction)
        
        # Connect to end
        self.add_edge("translate_to_english", END)
        self.add_edge("translate_to_user_language", END)
    
    @property
    def workflow_name(self):
        return "translation_workflow"
    
    def run(self):
        """
        Runs the translation workflow and returns the translated text.
        """
        logger.info("--- Starting Translation workflow ---")
        
        # Run the graph with the current state
        events = self.graph_streaming(self.state, recursion_limit=10)
        subfolder = self.write_events_to_file(events, self.output_filename)
        
        # The state should now contain the translated text from the executed node
        translated_text = self.state.get("translated_text", "Translation failed")
        
        logger.info(f"--- Translation workflow completed. Result: {translated_text} ---")
        
        return translated_text


# Custom subgraph classes that use the fixed message functions
class CustomTranslatorEnSubgraf(TranslatorEnSubgraf):
    def run(self, extra_input=None):
        logger.info("--- Starting CustomTranslatorEnSubgraf workflow ---")
        events = self.graph_streaming(extra_input or {}, recursion_limit=10)
        subfolder = self.write_events_to_file(events, self.output_filename)
        message = message_translator_en_fixed(self.state.get("language"), subfolder, self.output_filename)
        logger.info("--- CustomTranslatorEnSubgraf workflow completed ---")
        return message


class CustomTranslatorUsrSubgraf(TranslatorUsrSubgraf):
    def run(self, extra_input=None):
        logger.info("--- Starting CustomTranslatorUsrSubgraf workflow ---")
        events = self.graph_streaming(extra_input or {}, recursion_limit=10)
        subfolder = self.write_events_to_file(events, self.output_filename)
        message = message_translator_usr_fixed(self.state.get("language"), subfolder, self.output_filename)
        logger.info("--- CustomTranslatorUsrSubgraf workflow completed ---")
        return message


def main():
    """
    Demonstrate the translation workflow.
    """
    print("=" * 60)
    print("Black LangCube - Translation Workflow Example")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("./translation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Translate French to English
    print("\n1. Translating French to English:")
    print("-" * 40)
    
    french_text = "Bonjour le monde! Comment allez-vous aujourd'hui?"
    
    workflow1 = TranslationWorkflow(
        source_text=french_text,
        source_language="French",
        target_language="English",
        folder_name=str(output_dir / "french_to_english")
    )
    
    try:
        result1 = workflow1.run()
        print(f"Source (French): {french_text}")
        print(f"Translation: {result1}")
        print(f"Tokens used: {workflow1.state.get('translation_tokens', {})}")
    except Exception as e:
        print(f"Error in French to English translation: {e}")
    
    # Example 2: Translate English to Spanish
    print("\n2. Translating English to Spanish:")
    print("-" * 40)
    
    english_text = "Hello world! How are you doing today?"
    
    workflow2 = TranslationWorkflow(
        source_text=english_text,
        source_language="English",
        target_language="Spanish",
        folder_name=str(output_dir / "english_to_spanish")
    )
    
    try:
        result2 = workflow2.run()
        print(f"Source (English): {english_text}")
        print(f"Translation: {result2}")
        print(f"Tokens used: {workflow2.state.get('translation_tokens', {})}")
    except Exception as e:
        print(f"Error in English to Spanish translation: {e}")
    
    print("\n" + "=" * 60)
    print("Translation workflow examples completed!")


if __name__ == "__main__":
    main()
