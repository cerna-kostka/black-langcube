# Black LangCube Examples with Dummy Implementations

This directory contains working examples of how to use Black LangCube as a library, using dummy/mock implementations instead of the complex actual graph implementations.

## Files Created

### Dummy Implementations
- `dummy_implementations.py` - Contains mock classes for all the graph components (reference only, not used in final version)

### Modified Core Files
The following core files were replaced with dummy implementations:
- `../graf/graf1.py` - Mock Graph1 for question processing
- `../graf/graf2.py` - Mock Graph2 for keyword processing  
- `../graf/graf3.py` - Mock Graph3 for strategy generation
- `../graf/graf4.py` - Mock Graph4 for search and analysis
- `../graf/graf5.py` - Mock Graph5 for final processing

### Example Usage
- `library_usage.py` - Complete example showing how to use the library with various workflows and pipelines

## How It Works

The dummy implementations provide realistic mock data and simulate the actual workflow processing without requiring:
- Complex LLM integrations
- External API calls
- Heavy dependencies
- Actual analysis logic

Each mock graph:
1. Simulates processing time with `time.sleep(0.1)`
2. Creates output directories
3. Generates realistic mock results with relevant data
4. Saves results to JSON files
5. Returns structured data that matches the expected interface

## Running the Examples

```bash
# Activate the virtual environment
source ../.venv/bin/activate

# Run the complete library usage example
python library_usage.py
```

## What You'll See

The examples demonstrate:
1. **Single Workflow Execution** - Running individual graph workflows
2. **Complete Pipeline** - Running multiple workflows in sequence
3. **Error Handling** - How the library handles invalid inputs
4. **Session Management** - Creating and cleaning up processing sessions
5. **Library Integration** - How to integrate Black LangCube into larger applications

## Output

The examples create several output directories:
- `library_output/` - Contains results from the main examples
- `app_output/` - Contains results from the application integration example

Each workflow generates JSON files with mock analysis results, demonstrating the structure and type of data the real system would produce.

## Benefits of This Approach

- ✅ **Fast execution** - No waiting for LLM responses
- ✅ **No dependencies** - Works without external APIs or complex setups
- ✅ **Predictable results** - Same output every time for testing
- ✅ **Educational** - Shows the interface and data flow clearly
- ✅ **Development friendly** - Easy to modify and extend

This setup allows you to understand and work with the Black LangCube library interface without needing the full complexity of the actual implementation.
