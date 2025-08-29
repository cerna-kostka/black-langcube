# Black LangCube Examples

This directory contains example scripts demonstrating how to use the Black LangCube library.

## Prerequisites

1. Install the library:
   ```bash
   cd ..
   pip install -e .
   ```

2. Set up environment variables in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LANGCHAIN_API_KEY=your_langchain_api_key_here
   LANGCHAIN_TRACING_V2=true
   ```

## Examples

### 1. Basic Workflow (`basic_workflow.py`)
Demonstrates the fundamental concepts of creating a workflow with the BaseGraph class.

- **What it shows**: Basic graph construction, node definition, edge connections
- **Run**: `python basic_workflow.py`

### 2. Translation Workflow (`translation_workflow.py`)
Shows how to use the built-in translation subgraphs for multi-language processing.

- **What it shows**: Subgraph usage, conditional routing, translation pipelines
- **Run**: `python translation_workflow.py`

### 3. Custom LLM Node Example (`custom_llm_node.py`)
Demonstrates how to create custom LLM nodes for specific tasks.

- **What it shows**: Custom node creation, LLM integration, prompt engineering
- **Run**: `python custom_llm_node.py`

### 4. Data Structures Example (`data_structures_example.py`)
Shows how to use and extend the built-in Pydantic data structures.

- **What it shows**: Data validation, structured outputs, scientific article processing
- **Run**: `python data_structures_example.py`

## Notes

- All examples create output directories to store intermediate and final results
- Make sure you have sufficient OpenAI API credits for the LLM-based examples
- The examples are designed to be educational and may not represent production-ready code
- Feel free to modify and experiment with the examples to understand the library better

## Troubleshooting

If you encounter import errors, make sure:
1. The library is installed in your current Python environment
2. You're running from the correct directory
3. All dependencies are installed (see `requirements.txt` in the parent directory)

For more complex usage patterns, see the test files and the library's internal graph implementations in the `graf/` directory.
