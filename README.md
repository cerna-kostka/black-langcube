# Black LangCube

A LangGraph-based extension framework designed to facilitate the development of complex applications by providing a structured way to define and manage workflows.

## 🚀 Features

- **BaseGraph Framework**: Foundational interface for constructing, compiling, and executing stateful workflow graphs
- **Data Structures**: Pydantic models for scientific article metadata, search strategies, outlines, and more
- **LLM Nodes**: Pre-built nodes for common language model operations
- **Helper Utilities**: Token counting, result processing, file management, and workflow utilities
- **Subgraph System**: Modular subworkflows for translation, output generation, and specialized tasks
- **Extensible Architecture**: Easy to extend with custom nodes and workflows

## 📦 Installation

### From PyPI (when published):
```bash
pip install black_langcube
```

### Development Installation:
```bash
git clone https://github.com/your-username/black-langcube.git # Replace the placeholder
cd black-langcube
pip install -e .
```

### With optional dependencies:
```bash
pip install black_langcube[dev,examples]
```

## 🏗️ Core Components

### BaseGraph
The foundation for building stateful workflow graphs using LangGraph:

```python
from black_langcube.graf.graph_base import BaseGraph, GraphState

class MyCustomGraph(BaseGraph):
    def __init__(self, user_message, folder_name, language):
        super().__init__(MyGraphState, user_message, folder_name, language)
        self.build_graph()
    
    def build_graph(self):
        # Add nodes and edges to your workflow
        self.add_node("my_node", my_node_function)
        self.add_edge(START, "my_node")
        self.add_edge("my_node", END)
    
    @property
    def workflow_name(self):
        return "my_custom_graph"
```

### LLMNode
A base class for defining nodes that interact with language models:

```python
from black_langcube.llm_modules.LLMNodes.LLMNode import LLMNode

class MyCustomNode(LLMNode):
    def generate_messages(self):
        return [
            ("system", "You are a helpful assistant"),
            ("human", self.state.get("user_input", ""))
        ]

    def execute(self, extra_input=None):
        result, tokens = self.run_chain(extra_input)
        return {"output": result, "tokens": tokens}
```

### Data Structures
Pydantic models for structured data handling:

```python
from black_langcube.data_structures.data_structures import Article, Strategies, Outline

# Use pre-defined data structures
article = Article(topic="AI Research", language="English")
strategies = Strategies(strategy1="Search academic papers", strategy2="Analyze trends")
```

### LLM Nodes
Pre-built nodes for language model operations:

```python
from black_langcube.llm_modules.LLMNodes.LLMNode import LLMNode

class MyCustomNode(LLMNode):
    def generate_messages(self):
        return [
            ("system", "You are a helpful assistant"),
            ("human", self.state.get("user_input", ""))
        ]
    
    def execute(self, extra_input=None):
        result, tokens = self.run_chain(extra_input)
        return {"output": result, "tokens": tokens}
```

## 📚 Architecture

The library is organized into several key modules:

- **`graf/`**: Core graph classes and workflow definitions
- **`data_structures/`**: Pydantic models for data validation
- **`llm_modules/`**: Language model integration and node definitions
- **`helper_modules/`**: Utility functions and helper classes
- **`messages/`**: Message formatting and composition utilities
- **`prompts/`**: Prompt templates and configurations
- **`format_instructions/`**: Output formatting utilities

## 🛠️ Usage Examples

### Basic Workflow

```python
from black_langcube.graf.graph_base import BaseGraph, GraphState
from langgraph.graph import START, END

class SimpleWorkflow(BaseGraph):
    def __init__(self, message, folder, language):
        super().__init__(GraphState, message, folder, language)
        self.build_graph()
    
    def build_graph(self):
        def process_message(state):
            return {"result": f"Processed: {state['messages'][-1].content}"}
        
        self.add_node("process", process_message)
        self.add_edge(START, "process")
        self.add_edge("process", END)
    
    @property
    def workflow_name(self):
        return "simple_workflow"

# Usage
workflow = SimpleWorkflow("Hello, world!", "output", "English")
result = workflow.run()
```

### Using Subgraphs

```python
from black_langcube.graf.subgrafs.translator_en_subgraf import TranslatorEnSubgraf

# Translation subgraph
translator = TranslatorEnSubgraf(config, subfolder="translations")
result = translator.run(extra_input={
    "translation_input": "Bonjour le monde",
    "language": "French"
})
```

## 🔧 Configuration

The library uses environment variables for configuration. Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here

# optional: LangChain configuration
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
```

## 📖 Examples

See the `examples/` directory for complete working examples:

- **Basic Graph**: Simple workflow with custom nodes
- **Translation Pipeline**: Multi-language processing workflow
- **Scientific Article Processing**: Complex multi-step analysis pipeline
- **Custom Data Structures**: Extending the framework with your own models

## 🧪 Development

### Setting up development environment:

```bash
git clone https://github.com/your-username/black-langcube.git
cd black-langcube
pip install -e .[dev]
```

### Running tests:

```bash
pytest
```

### Code formatting:

```bash
black .
isort .
```

## 📋 Requirements

- Python 3.9+
- LangChain >= 0.3.24
- LangGraph >= 0.3.7
- Pydantic >= 2.0.0
- OpenAI API access

## 🤝 Contributing

This is a work in progress and contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License (MIT) 

## ⚠️ Note

This library is intended to be used within a larger application context. The code is provided as-is and is actively being improved. Take it with a grain of salt and feel free to contribute improvements!

## 🔗 Links

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Examples and Tutorials](./examples/)