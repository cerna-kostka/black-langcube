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
│   ├── database/                 # SQLAlchemy async ORM storage layer
│   │   ├── __init__.py           # Exports DatabaseService, Base, ORM models
│   │   ├── config.py             # Async engine, session factory, Base
│   │   ├── models.py             # Session, GraphOutput, NodeOutput, TokenUsage ORM models
│   │   └── operations.py        # DatabaseService context manager + sanitize_json_data
│   ├── storage_service.py        # StorageService (file / database / dual modes)
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

### 4. Database Storage Layer (`database/`)

SQLAlchemy 2.x async ORM storage layer that replaces (or augments) the
file-based JSON output system.

#### Subpackage files

| File | Purpose |
|---|---|
| `config.py` | Async engine factory, session factory (`async_sessionmaker`), `Base` |
| `models.py` | ORM models: `Session`, `GraphOutput`, `NodeOutput`, `TokenUsage` |
| `operations.py` | `DatabaseService` context manager + `sanitize_json_data()` |

#### Storage modes

Controlled by the `STORAGE_MODE` environment variable:

| Mode | Behavior |
|---|---|
| `file` | Write/read files only — default, fully backward-compatible |
| `database` | Write/read database only |
| `dual` | Write to both; read from database — recommended migration path |

Unknown mode values cause `StorageService` to raise `ValueError` at
construction time (fail fast, not at write time).

#### Database URL

Set `DATABASE_URL` to your connection string. The library auto-converts it to
the appropriate async dialect:

```
# PostgreSQL (production)
DATABASE_URL=postgresql://user:password@host:5432/dbname
# → postgresql+asyncpg://...

# SQLite (local / tests)
DATABASE_URL=sqlite:///./black_langcube.db
# → sqlite+aiosqlite://...
```

#### Running DB tests

Install the optional database group and run pytest:

```bash
pip install -e .[dev,database]
pytest tests/test_database.py -v
```

All DB tests use an in-memory SQLite database (`sqlite+aiosqlite:///:memory:`
with `StaticPool`). The `DATABASE_URL` override is applied in `conftest.py`
before any `black_langcube.database` imports so the engine is created with the
test URL.
- `Article`: Scientific article metadata
- `Outline`: Document structure representation
- Extensible for custom data types

### 5. Processing Entry Points (`process.py`)
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

3. **Validate Configuration**:
   Consuming applications should call `validate_config()` at startup to detect
   misconfiguration before any pipeline runs:

   ```python
   from black_langcube import validate_config, ConfigurationError
   import sys

   try:
       validate_config()
   except ConfigurationError as e:
       print(f"Configuration error: {e}", file=sys.stderr)
       sys.exit(1)
   ```

   `validate_config()` checks all entries in `config.REQUIRED_ENV_VARS` and
   raises `ConfigurationError` listing every missing variable in one message.
   The library itself never calls `sys.exit()` — error handling is the
   caller's responsibility.

   Required variables are read via `get_api_key(env_var)`, which wraps the
   value in `pydantic.SecretStr` to prevent accidental leakage in logs or
   tracebacks. The raw value is accessible only via `.get_secret_value()` and
   should be retrieved only when actually passed to an external client.

4. **Run Examples**:
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

## Running Tests

### Installing Test Dependencies

Install the `dev` and `database` optional groups before running tests. `aiosqlite`
and `pytest-asyncio` are included in the `dev` group:

```bash
pip install -e .[dev,database]
```

### Fixture Setup

`tests/conftest.py` bootstraps the test environment automatically:

- `DATABASE_URL=sqlite+aiosqlite:///:memory:` is set via `os.environ.setdefault`
  **before** any `black_langcube.database` imports so the in-memory engine is
  created with the test URL.
- `OPENAI_API_KEY=test-key-not-real` is set so that config-level validation
  passes during test collection on machines without a real API key.  Real keys
  in the environment always take precedence (`setdefault` is used throughout).
- `asyncio_mode = "auto"` (in `pyproject.toml`) means every `async def test_*`
  function runs automatically under pytest-asyncio — no per-function decorator
  needed.

### Marker-Based Test Selection

All tests are tagged with one of three markers registered in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `unit` | Fast isolated tests — no external services, no file I/O |
| `integration` | Tests using in-memory SQLite, `tmp_path`, or multiple internal components |
| `functional` | End-to-end tests requiring a running application or live external service |

Run a specific subset:

```bash
# Only unit tests (fastest — no DB, no file I/O)
pytest -m unit

# Only integration tests
pytest -m integration

# Everything except functional (safe for CI without live services)
pytest -m "not functional"

# Full suite
pytest
```

### Using the `mock_robust_invoke` Fixture

`mock_robust_invoke` is an opt-in fixture (not `autouse`) that patches
`robust_invoke_async` at the module level so no real API calls are made.

```python
async def test_my_node_output(mock_robust_invoke):
    # Configure the return value the mock should produce
    mock_robust_invoke.return_value = {"output": "mocked response", "tokens": {}}

    # Run the code under test
    result = await my_node(state={"question": "test?"}, extra_input={})

    assert result["output"] == "mocked response"
    # Confirm the mock was actually called
    mock_robust_invoke.assert_awaited_once()
```

The mock is an `AsyncMock`, so `await`ing it works naturally and call history
is tracked as expected.

### The `test_db_session` Fixture

`test_db_session` provides a clean `AsyncSession` backed by an in-memory SQLite
database for every test.  All writes are rolled back in the `finally` block —
no explicit cleanup is required.

```python
async def test_creates_row(test_db_session):
    from black_langcube.database.models import Session as SessionModel
    import uuid

    row = SessionModel(id=str(uuid.uuid4()), status="running")
    test_db_session.add(row)
    await test_db_session.flush()

    assert row.id is not None
    # Rollback happens automatically after this test
```

### `@pytest.mark.skip` Discipline

Any test that genuinely requires a live external service (real API key, running
Postgres, etc.) must be decorated explicitly — never silently skip by catching
a missing-dependency error:

```python
@pytest.mark.skip(reason="Requires live OpenAI API key")
async def test_live_llm_call():
    ...
```

This makes it obvious in CI output which tests were omitted and why.

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

## Logging

### Library logging rule

Every module that emits log messages must obtain its logger with:

```python
import logging
logger = logging.getLogger(__name__)
```

Library modules must **never** call `logging.basicConfig()` or add handlers to the root logger.
That is the sole responsibility of the consuming application.
A consumer can silence all routine library output with one line:

```python
logging.getLogger("black_langcube").setLevel(logging.WARNING)
```

Example scripts in `src/black_langcube/examples/` are entry-point scripts and may configure
logging for demonstration purposes — this is expected and intentional.

### Severity policy

| Level | Use for |
|---|---|
| `DEBUG` | Routine per-invocation detail: node entry/exit banners, retry sleep messages, token counts, individual streaming events |
| `INFO` | Successful high-level state transitions: graph completed, session folder created |
| `WARNING` | Non-fatal unexpected conditions: missing optional config, fallback triggered, empty result list |
| `ERROR` | Recoverable failures: file not found, invalid JSON line, API error after all retries, language key missing |
| `CRITICAL` | Unrecoverable / process-threatening failures only (use sparingly; currently only "Failed to create session folder") |
