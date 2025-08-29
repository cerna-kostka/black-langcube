# Black LangCube Development Guide

## Project Structure

The Black LangCube library is organized as follows:

```
black-langcube/
├── src/black_langcube/           # Source code in src layout
│   ├── __init__.py               # Main library entry point
│   ├── process.py                # Core processing functions
│   ├── graf/                     # Core graph classes and workflows
│   │   ├── __init__.py
│   │   ├── graph_base.py         # BaseGraph foundation class
│   │   ├── graf1.py              # Question processing workflow
│   │   ├── graf2.py              # Keyword processing workflow  
│   │   ├── graf3.py              # Strategy generation workflow
│   │   ├── graf4.py              # Search and analysis workflow
│   │   ├── graf5.py              # Final processing workflow
│   │   └── subgrafs/             # Modular subworkflows
│   │       ├── message_translator_subgraf.py
│   │       ├── translator_en_subgraf.py
│   │       └── translator_usr_subgraf.py
│   ├── data_structures/          # Pydantic data models
│   ├── llm_modules/              # Language model integration
│   │   ├── LLMNodes/            # Pre-built LLM node classes
│   │   │   ├── LLMNode.py       # Base LLM node class
│   │   │   └── subgraphs/       # LLM subgraph implementations
│   │   ├── llm_model.py         # Model configurations
│   │   ├── parsers.py           # Output parsers
│   │   ├── check_tokens.py      # Token validation utilities
│   │   ├── CheckTitleRelevance.py # Title relevance checker
│   │   └── robust_invoke.py     # Error handling
│   ├── helper_modules/           # Utility functions and helpers
│   │   ├── helper_nodes/        # Helper node implementations
│   │   ├── token_counter/       # Token counting utilities
│   │   ├── calculate_duration.py
│   │   ├── escaping.py
│   │   ├── get_basegraph_classes.py
│   │   ├── get_result_from_graph_outputs.py
│   │   └── submodules.py
│   ├── messages/                 # Message formatting and composition
│   │   └── subgraphs/           # Message subgraph implementations
│   ├── prompts/                  # Prompt templates
│   ├── format_instructions/      # Output formatting utilities
│   │   └── analyst_format.py    # Analyst-specific formatting
│   ├── output_creation_functions/ # File generation utilities
│   └── examples/                 # Usage examples and tutorials
├── tests/                        # Test files
├── pyproject.toml               # Modern Python packaging configuration
├── requirements.txt             # Core dependencies
├── LICENSE                      # MIT License
├── README.md                    # Main documentation
└── DEVELOPMENT.md               # This development guide
```

## Key Components

### 1. BaseGraph (`graf/graph_base.py`)
The foundation class for all workflow graphs. Provides:
- Graph construction methods (`add_node`, `add_edge`, `add_conditional_edges`)
- State management and serialization
- Event streaming and file output
- Compilation and execution framework

### 2. LLM Nodes (`llm_modules/LLMNodes/`)
Pre-built node classes for common LLM operations:
- `LLMNode`: Base class for language model nodes
- Specialized nodes for translation, analysis, generation
- Error handling and token counting

### 3. Data Structures (`data_structures/`)
Pydantic models for structured data:
- `Strategies`: Research strategy definitions
- `Article`: Scientific article metadata
- `Outline`: Document structure representation
- Extensible for custom data types

### 4. Processing Entry Points (`process.py`)
Main library interface functions:
- `run_workflow_by_id()`: Execute individual workflows
- `run_complete_pipeline()`: Run sequential workflow chains
- `cleanup_session()`: Session management

## Development Workflow

### Setting up Development Environment

1. **Clone and Install**:
   ```bash
   git clone <repository-url>
   cd black-langcube
   pip install -e .
   ```

2. **Environment Variables**:
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here
   LANGCHAIN_TRACING_V2=true
   ```

3. **Run Examples**:
   ```bash
   cd examples
   python basic_workflow.py
   python library_usage.py
   ```

### Adding New Workflows

1. **Create Graph Class**:
   ```python
   # src/black_langcube/graf/my_workflow.py
   from .graph_base import BaseGraph, GraphState
   
   class MyWorkflowState(GraphState):
       # Define your state fields
       input_data: str
       output_data: str
   
   class MyWorkflow(BaseGraph):
       def build_graph(self):
           # Define your workflow nodes and edges
           pass
   ```

2. **Add Processing Function**:
   ```python
   # In src/black_langcube/process.py
   def _run_my_workflow(user_message, folder_name, language):
       workflow = MyWorkflow(user_message, folder_name, language)
       return workflow.run()
   ```

3. **Update Main Router**:
   ```python
   # Add case to run_workflow_by_id()
   elif workflow_id == "my_workflow":
       return _run_my_workflow(user_message, folder_name, language)
   ```

### Adding Custom Nodes

1. **Inherit from LLMNode**:
   ```python
   from llm_modules.LLMNodes.LLMNode import LLMNode
   
   class MyCustomNode(LLMNode):
       def generate_messages(self):
           # Return list of (role, content) tuples
           return [("system", "System prompt"), ("human", "User input")]
       
       def execute(self, extra_input=None):
           # Implement your node logic
           result, tokens = self.run_chain(extra_input)
           return {"output": result, "tokens": tokens}
   ```

2. **Use in Workflows**:
   ```python
   def build_graph(self):
       my_node = MyCustomNode(self.state, self.config)
       self.add_node("my_node", my_node.execute)
   ```

### Testing

Run examples to test functionality:
```bash
cd src/black_langcube/examples
python basic_workflow.py
python translation_workflow.py
python custom_llm_node.py
python data_structures_example.py
python library_usage.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add examples if applicable
5. Test your changes
6. Submit a pull request

## Architecture Principles

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new workflows and nodes
3. **Type Safety**: Pydantic models for data validation
4. **Error Handling**: Robust error handling and logging
5. **Documentation**: Clear docstrings and examples
6. **Backward Compatibility**: Maintain stability

## Common Patterns

### State Management
```python
class MyState(GraphState):
    # Define typed fields
    data: str
    processed: bool = False

# Access in nodes
def my_node(state):
    data = state.get("data", "")
    state["processed"] = True
    return {"result": f"Processed: {data}"}
```

### Conditional Routing
```python
def route_condition(state):
    if state.get("condition"):
        return "path_a"
    return "path_b"

self.add_conditional_edges(
    "decision_node",
    route_condition,
    {"path_a": "node_a", "path_b": "node_b"}
)
```

### Subgraph Integration
```python
from graf.subgrafs.translator_en_subgraf import TranslatorEnSubgraf

def my_node(state):
    subgraf = TranslatorEnSubgraf(config, subfolder)
    result = subgraf.run(extra_input={"data": state["data"]})
    return {"subgraf_result": result}
```
