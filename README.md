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
git clone https://github.com/cerna-kostka/black-langcube.git
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
- **`database/`**: SQLAlchemy async ORM models and `DatabaseService`
- **`storage_service.py`**: Three-mode storage abstraction (`file`, `database`, `dual`)

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

The library uses environment variables for configuration. Copy `.env.example`
from the project root to `.env` and fill in your values — it documents every
configurable variable with its default and a one-line description.

```env
OPENAI_API_KEY=your_openai_api_key_here

# optional: LangChain configuration
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
```

### LLM Configuration

#### Global provider

Set `PROVIDER` to choose the default LLM provider for every processing step:

```env
PROVIDER=openai   # openai (default) | gemini | mistral
```

#### Per-step provider overrides

Each pipeline step can use a different provider without changing any code.
Set `{STEP}_PROVIDER` to override only that step; all other steps continue to
use `PROVIDER`:

| Step | Override variable | Example |
|------|-------------------|---------|
| `llm_analyst()` | `ANALYST_PROVIDER` | `ANALYST_PROVIDER=gemini` |
| `llm_outline()` | `OUTLINE_PROVIDER` | `OUTLINE_PROVIDER=openai` |
| `llm_text()` | `TEXT_PROVIDER` | `TEXT_PROVIDER=gemini` |
| `llm_check_title()` | `CHECK_TITLE_PROVIDER` | `CHECK_TITLE_PROVIDER=openai` |
| `llm_title_abstract()` | `TITLE_ABSTRACT_PROVIDER` | `TITLE_ABSTRACT_PROVIDER=openai` |
| `get_llm_low()` | `LOW_PROVIDER` | `LOW_PROVIDER=mistral` |
| `get_llm_high()` | `HIGH_PROVIDER` | `HIGH_PROVIDER=openai` |

#### Per-step model name overrides

Override the model name for a specific `(provider, step)` combination using
`{PROVIDER}_MODEL_{STEP}`:

```env
OPENAI_MODEL_LOW=gpt-4o-mini        # default
OPENAI_MODEL_HIGH=gpt-4.1           # default
GEMINI_MODEL_ANALYST=gemini-2.5-pro # default
GEMINI_MODEL_CHECK_TITLE=gemini-2.5-flash  # use cheaper model for title checks
```

> **Note:** `{STEP}_PROVIDER` overrides are read on every factory call and take
> effect immediately without a restart. `{PROVIDER}_MODEL_{STEP}` overrides are
> evaluated once at module import time — a process restart is required for
> changes to model-name env vars to take effect.

#### Mixed-provider example

Use Gemini for cost-sensitive steps and OpenAI for quality-critical ones
without any code changes:

```env
PROVIDER=openai                  # default for all unspecified steps

ANALYST_PROVIDER=gemini          # cost-sensitive analysis
TEXT_PROVIDER=gemini             # cost-sensitive text generation
OUTLINE_PROVIDER=openai          # quality-critical outline
CHECK_TITLE_PROVIDER=openai      # quality-critical title check

GEMINI_API_KEY=your-gemini-key-here
OPENAI_API_KEY=your-openai-key-here
```

#### Verifying the resolved configuration

Use `get_llm_config_summary()` to print the resolved `(provider, model)` for
every step — useful at startup or in test logs:

```python
from black_langcube import get_llm_config_summary

summary = get_llm_config_summary()
for step, info in summary.items():
    print(f"{step:20s} provider={info['provider']}  model={info['model']}")
```

Example output with the mixed-provider configuration above:

```
analyst              provider=gemini  model=gemini-2.5-pro
outline              provider=openai  model=gpt-4.1
text                 provider=gemini  model=gemini-2.5-pro
check_title          provider=openai  model=gpt-4.1
title_abstract       provider=openai  model=gpt-4.1
low                  provider=openai  model=gpt-4o-mini
high                 provider=openai  model=gpt-4.1
```

#### Optional provider dependencies

The default `pip install black_langcube` includes only the OpenAI integration.
Install additional extras for other providers:

```bash
pip install black_langcube[gemini]   # adds langchain-google-genai
pip install black_langcube[mistral]  # adds langchain-mistralai
```

### Fail-Fast Validation

Call `validate_config()` at the top of your application entry point to detect
misconfiguration immediately, before any pipeline execution begins:

```python
from black_langcube import validate_config, ConfigurationError
import sys

try:
    validate_config()
except ConfigurationError as e:
    print(f"Configuration error: {e}", file=sys.stderr)
    sys.exit(1)
```

`validate_config()` checks every required environment variable and raises
`ConfigurationError` with a message listing **all** missing variables, so you
see every problem at once. It is safe to call multiple times (idempotent).

API keys are stored internally as `pydantic.SecretStr`, which prevents the raw
value from appearing in `str()`, `repr()`, or log output. Call
`.get_secret_value()` only at the last moment when the key must be used.

### Storage and Database Configuration

The library supports three output storage modes controlled by the `STORAGE_MODE`
environment variable:

| `STORAGE_MODE` | Behavior |
|---|---|
| `file` (default) | Write results to timestamped folders — existing behavior, fully backward-compatible |
| `database` | Write results only to the database |
| `dual` | Write to both file system and database — recommended migration path |

Set a database connection URL via the `DATABASE_URL` environment variable:

```env
# SQLite (local/testing)
DATABASE_URL=sqlite:///./black_langcube.db

# PostgreSQL (production)
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

The library automatically converts `DATABASE_URL` to the appropriate async
dialect (`postgresql+asyncpg://` or `sqlite+aiosqlite://`).

#### Optional database dependencies

Install the `database` extras to enable database-backed storage:

```bash
pip install black_langcube[database]
```

This installs `sqlalchemy[asyncio]>=2.0`, `asyncpg` (PostgreSQL), and
`aiosqlite` (SQLite / tests).

#### Migration guide for existing `file`-mode users

Existing deployments are **unaffected by default**. `STORAGE_MODE` defaults to
`file` when the environment variable is unset. To migrate:

1. Install `black_langcube[database]`.
2. Set `DATABASE_URL` to your database connection string.
3. Start with `STORAGE_MODE=dual` to write to both file and database while you
   verify the database output.
4. Switch to `STORAGE_MODE=database` once you are satisfied.

#### Using `StorageService` directly

```python
import asyncio
from black_langcube.storage_service import StorageService

async def main():
    # Uses STORAGE_MODE and DATABASE_URL from environment
    storage = StorageService()
    await storage.save_graph_output(
        session_id="my-session-uuid",
        graph_name="graf1",
        data={"result": "..."},
        step_name="analysis",
    )

asyncio.run(main())
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
git clone https://github.com/cerna-kostka/black-langcube.git
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

## Parallel Fan-Out (Scatter-Gather)

`BaseGraph` exposes `add_parallel_nodes` for wiring an intra-graph fan-out: a single node dispatches to multiple branches that run **concurrently** (via LangGraph's `Send` API), and a merge node aggregates their results.

### State setup

Use `operator.add` (or any reducer) with `Annotated` so that concurrent branches can each append to the same list field without overwriting each other:

```python
import operator
from typing import Annotated
from black_langcube.graf.graph_base import GraphState

class FanOutState(GraphState):
    topic: str
    branch_results: Annotated[list, operator.add]  # reducer – each branch appends
    merged_summary: str
```

### Graph wiring

```python
from langgraph.graph import START, END
from black_langcube.graf.graph_base import BaseGraph

class MyFanOutGraph(BaseGraph):
    def __init__(self, topic, folder, language="English"):
        super().__init__(FanOutState, topic, folder, language)
        self._topic = topic
        self._build()

    def _build(self):
        def prepare(state):
            return {}                                   # fan-out source

        def branch_a(state):
            return {"branch_results": [f"A: {state['topic']}"]}

        def branch_b(state):
            return {"branch_results": [f"B: {state['topic']}"]}

        def merge(state):
            return {"merged_summary": " | ".join(state["branch_results"])}

        self.add_node("prepare", prepare)
        self.add_node("branch_a", branch_a)
        self.add_node("branch_b", branch_b)
        self.add_node("merge", merge)

        self.add_edge(START, "prepare")
        # Wire fan-out → concurrent branches → merge
        self.add_parallel_nodes("prepare", ["branch_a", "branch_b"], "merge")
        self.add_edge("merge", END)

    @property
    def workflow_name(self):
        return "my_fanout"
```

A custom `router_fn` can be supplied to control what state each branch
receives:

```python
from langgraph.types import Send

def router(state):
    return [
        Send("branch_a", {**state, "mode": "fast"}),
        Send("branch_b", {**state, "mode": "thorough"}),
    ]

self.add_parallel_nodes("prepare", ["branch_a", "branch_b"], "merge", router_fn=router)
```

### Pipeline-level parallelism

To run **independent** graph instances simultaneously, use `run_parallel_pipeline`:

```python
import asyncio
from black_langcube import run_parallel_pipeline

graph_a = MyFanOutGraph("topic A", "output/a")
graph_b = MyFanOutGraph("topic B", "output/b")

results = asyncio.run(run_parallel_pipeline([graph_a, graph_b]))
# results["status"]           → "completed" | "partial_failure"
# results["parallel_results"] → [result_a, result_b]
```

See `src/black_langcube/examples/parallel_fanout_workflow.py` for a fully working end-to-end example.

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
- [Examples and Tutorials](./src/black_langcube/examples/)