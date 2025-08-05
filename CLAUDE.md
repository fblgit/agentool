# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
When exploring new tasks at any point where two different frameworks interconnect, you should prefer on searching the context7 MCP for documentation/code snippets/examples/etc.  

## Commands

### Testing
- Run all tests: `pytest`
- Run specific test file: `pytest tests/test_agentool.py`
- Run tests with coverage: `pytest --cov=src --cov-report=html`
- Run tests in verbose mode: `pytest -v`
- Run async tests: `pytest -v tests/test_agentool_async_sync.py`

### Development
- Install dependencies for development: `pip install -e .`

## Architecture Overview

AgenTools extends Pydantic-AI with deterministic tool execution capabilities. The framework enables:

1. **AgenTool Components**: Core framework for creating deterministic agents that mimic LLM behavior while executing specific tools based on structured inputs.

2. **AgenToolkits**: Collection of pre-built tool collections for common tasks like auth, storage, HTTP, crypto, metrics, logging, etc.

3. **Workflow AgenToolkits**: LLM-powered agents for analyzing, specifying, crafting, and evaluating code using the AgenTool framework.

### Key Components

**Core AgenTool System** (`src/agentool/`):
- `base.py`: Base schema for all AgenTool inputs
- `core/model.py`: Synthetic LLM model that provides deterministic execution
- `core/manager.py`: Handles routing and payload transformation between schemas and tools
- `core/registry.py`: Global configuration store for AgenTool instances
- `core/injector.py`: Dependency injection for multi-agent systems
- `factory.py`: High-level functions for creating AgenTools

**AgenToolkits** (`src/agentoolkit/`):
- Pre-built collections of tools organized by domain (auth, storage, network, etc.)
- Each toolkit provides ready-to-use functionality that can be composed into agents

**Workflow System** (`src/agentoolkit/workflows/`):
- LLM-powered agents that help create new AgenTools
- Includes analyzer, specifier, crafter, and evaluator agents

### How AgenTools Work

1. **Creation**: Use `create_agentool()` to define an agent with:
   - Input schema (Pydantic model)
   - Routing configuration (maps operations to tools)
   - Tool functions (actual implementation)

2. **Execution**: When an agent runs:
   - JSON input is parsed and validated against schema
   - Operation field determines which tool to call
   - Arguments are transformed to match tool signature
   - Tool executes and returns result

3. **Integration**: AgenTools integrate seamlessly with pydantic-ai:
   - Appear as regular LLM models
   - Support the same agent interface
   - Can be composed and injected as dependencies

### Example Usage

```python
from agentool import create_agentool, BaseOperationInput, RoutingConfig
from typing import Literal

class StorageInput(BaseOperationInput):
    operation: Literal['read', 'write']
    key: str
    data: Optional[str] = None

storage_agent = create_agentool(
    name='storage',
    input_schema=StorageInput,
    routing_config=RoutingConfig(
        operation_field='operation',
        operation_map={
            'read': ('storage_read', lambda x: {'key': x.key}),
            'write': ('storage_write', lambda x: {'key': x.key, 'data': x.data}),
        }
    ),
    tools=[storage_read, storage_write]
)
```

### Testing Patterns

- Unit tests for individual AgenTools: `tests/test_agentool.py`
- Integration tests for multi-agent systems: `tests/test_agent_integration.py`
- Async/sync compatibility tests: `tests/test_agentool_async_sync.py`
- AgenToolkit-specific tests in `tests/agentoolkit/`

### Important Files

- `docs/architecture.md`: Comprehensive architecture guide
- `docs/CRAFTING_AGENTOOLS.md`: Guide for creating new AgenTools
- `examples/`: Example implementations and demos
- `src/templates/`: Jinja templates used by workflow agents
