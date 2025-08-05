# AgenTool - Deterministic Tool Execution for pydantic-ai

AgenTool is a framework that provides deterministic, schema-driven tool execution within [pydantic-ai](https://github.com/pydantic/pydantic-ai). It acts as a synthetic LLM model provider that routes JSON input to appropriate tools based on configuration, offering a cost-effective and predictable alternative to LLM calls for structured operations.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Use Cases](#use-cases)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)

## Overview

AgenTool bridges the gap between LLM-powered agents and programmatic tool execution. While pydantic-ai excels at creating LLM-based agents, there are scenarios where deterministic behavior is preferred:

- **Structured Operations**: CRUD operations, API calls, data transformations
- **Cost Optimization**: Avoiding LLM API costs for simple routing logic
- **Testing**: Predictable behavior for unit and integration tests
- **Hybrid Systems**: Mixing LLM intelligence with deterministic execution

## Key Features

- 🎯 **Schema-Driven**: Uses Pydantic models for input validation
- 🔀 **Smart Routing**: Routes operations to tools based on configurable fields
- 🔧 **Tool Integration**: Works with any async Python function as a tool
- 📦 **Zero Dependencies**: Only requires pydantic-ai (and thus pydantic)
- 🎭 **LLM Interface**: Maintains the same interface as LLM-based agents
- 💰 **Cost Effective**: No LLM API calls for deterministic operations
- 🧪 **Testable**: Predictable behavior makes testing straightforward

## Installation

```bash
# AgenTool is part of the pydantic-ai ecosystem
pip install pydantic-ai

# Then add the AgenTool module to your project
# (Currently available as a module, not a separate package)
```

## Project Structure

AgenTool cleanly separates core framework components from examples:
- **Core**: `agentool/core/` - The framework implementation
- **Examples**: `agentool/examples/` - Storage, compute, and other examples
- **Documentation**: See `STRUCTURE.md` for detailed organization

For pre-built examples, check the `examples/` directory.

## Quick Start

```python
from agentool import create_agentool, RoutingConfig
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from typing import Any

# 1. Define your input schema
class CalculatorInput(BaseModel):
    operation: str = Field(description="The operation to perform")
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

# 2. Create your tool functions
async def add(ctx: RunContext[Any], a: float, b: float) -> float:
    return a + b

async def multiply(ctx: RunContext[Any], a: float, b: float) -> float:
    return a * b

# 3. Configure routing
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'add': ('add', lambda x: {'a': x.a, 'b': x.b}),
        'multiply': ('multiply', lambda x: {'a': x.a, 'b': x.b})
    }
)

# 4. Create the AgenTool
calculator = create_agentool(
    name='calculator',
    input_schema=CalculatorInput,
    routing_config=routing,
    tools=[add, multiply]
)

# 5. Use it like any pydantic-ai agent
result = await calculator.run('{"operation": "add", "a": 5, "b": 3}')
print(result.output)  # "8.0"
```

## Architecture

AgenTool consists of four main components:

### 1. AgenToolModel

A synthetic LLM model provider that:
- Parses JSON input from user messages
- Generates tool calls to the routing manager
- Returns tool results as model responses

```python
from agentool import AgenToolModel

model = AgenToolModel('my_tool')
# Use with pydantic-ai: Agent(model=model)
```

### 2. AgenToolManager

The routing engine that:
- Validates input against schemas
- Routes operations to appropriate tools
- Transforms data between input schema and tool signatures
- Executes tools and returns results

### 3. AgenToolRegistry

Global configuration storage:
- Stores input schemas and routing configurations
- Enables dynamic tool lookup
- Supports multiple AgenTool instances

### 4. RoutingConfig

Defines how operations map to tools:
- Specifies the operation field name
- Maps operation values to tool names
- Defines data transformation functions

## Use Cases

### 1. API Gateway Pattern

```python
class APIRequest(BaseModel):
    endpoint: str
    method: str
    data: dict

routing = RoutingConfig(
    operation_field='endpoint',
    operation_map={
        'users': ('handle_users', lambda x: {'method': x.method, 'data': x.data}),
        'posts': ('handle_posts', lambda x: {'method': x.method, 'data': x.data}),
    }
)
```

### 2. Storage Operations

```python
# Example: Define your own schema or use one from examples
from pydantic import BaseModel
from typing import Literal

class StorageOperationInput(BaseModel):
    operation: Literal['read', 'write', 'list', 'delete']
    key: Optional[str] = None
    data: Optional[Any] = None

# Pre-built schema for storage operations
storage = create_agentool(
    name='storage',
    input_schema=StorageOperationInput,  # Includes read/write/list/delete
    routing_config=storage_routing,
    tools=[read_tool, write_tool, list_tool, delete_tool]
)
```

### 3. Workflow Orchestration

```python
class WorkflowInput(BaseModel):
    step: str
    context: dict

# Route to different workflow steps
workflow = create_agentool(
    name='workflow',
    input_schema=WorkflowInput,
    routing_config=workflow_routing,
    tools=[validate_step, process_step, notify_step]
)
```

### 4. Testing & Mocking

```python
# Replace LLM calls with deterministic behavior in tests
test_agent = create_agentool(
    name='test',
    input_schema=TestInput,
    routing_config=test_routing,
    tools=[mock_llm_response]
)
```

## API Reference

### create_agentool

```python
def create_agentool(
    name: str,
    input_schema: Type[BaseModel],
    routing_config: RoutingConfig,
    tools: List[Callable],
    system_prompt: Optional[str] = None,
    description: Optional[str] = None,
) -> Agent:
    """
    Factory function to create an AgenTool.
    
    Args:
        name: Unique identifier for the AgenTool
        input_schema: Pydantic model for input validation
        routing_config: Configuration for operation routing
        tools: List of async functions to use as tools
        system_prompt: Optional prompt for the agent
        description: Optional description
        
    Returns:
        A pydantic-ai Agent configured with AgenToolModel
    """
```

### RoutingConfig

```python
@dataclass
class RoutingConfig:
    """
    Configuration for routing operations to tools.
    
    Attributes:
        operation_field: Field name containing the operation (default: 'operation')
        operation_map: Dict mapping operation values to (tool_name, transform_func)
    """
    operation_field: str = 'operation'
    operation_map: Dict[str, Tuple[str, Callable[[Any], Dict[str, Any]]]] = None
```

### Base Schemas

AgenTool provides several pre-built schemas:

```python
# Base schema with just operation field
class BaseOperationInput(BaseModel):
    operation: str

# Storage operations
class StorageOperationInput(BaseOperationInput):
    operation: Literal['read', 'write', 'list', 'delete']
    key: Optional[str]
    data: Optional[Any]

# Compute operations  
class ComputeOperationInput(BaseOperationInput):
    operation: Literal['add', 'subtract', 'multiply', 'divide']
    a: float
    b: float

# Text operations
class TextOperationInput(BaseOperationInput):
    operation: Literal['concat', 'split', 'replace', 'format']
    text: Optional[str]
    pattern: Optional[str]
    replacement: Optional[str]
```

## Examples

### Decorator-Based Math Assistant

For a modern approach using pydantic-ai's decorator pattern with automatic schema extraction:

```python
from pydantic_ai import Agent, RunContext
from agentool import AgenToolModel, register_agentool_models

# Register AgenTool models
register_agentool_models()

# Create agent with AgenTool model
math_agent = Agent(
    model=AgenToolModel('math'),
    system_prompt="Mathematical assistant"
)

# Define tools with automatic schema extraction
@math_agent.tool(docstring_format='google')
async def calculate(ctx: RunContext[None], operation: MathOperation) -> dict:
    """Evaluate a mathematical expression.
    
    Args:
        operation: The math operation with expression and options
    """
    # Parse and evaluate expression
    # Return result as dict
    pass

# Create router for AgenTool
@math_agent.tool(name='__agentool_manager__')
async def router(ctx: RunContext[None], **kwargs) -> Any:
    return await calculate(ctx, MathOperation(**kwargs))

# Use it
result = await math_agent.run('{"expression": "10 + 5"}')
```

See `examples/demos/calculator_decorated.py` for the complete implementation with:
- Automatic schema generation from docstrings
- Step-by-step calculation explanations
- Error handling for edge cases
- Single-model input with simplified schema

### Complete Storage System

```python
from agentool import create_agentool, RoutingConfig
# Example: Define your own schema or use one from examples
from pydantic import BaseModel
from typing import Literal

class StorageOperationInput(BaseModel):
    operation: Literal['read', 'write', 'list', 'delete']
    key: Optional[str] = None
    data: Optional[Any] = None
from pydantic_ai import RunContext
from typing import Any, Dict

# Storage backend
storage_backend: Dict[str, Any] = {}

# Tool implementations
async def storage_read(ctx: RunContext[Any], key: str) -> Any:
    return storage_backend.get(key, f"Key not found: {key}")

async def storage_write(ctx: RunContext[Any], key: str, data: Any) -> str:
    storage_backend[key] = data
    return f"Stored data at key: {key}"

async def storage_list(ctx: RunContext[Any]) -> list[str]:
    return list(storage_backend.keys())

async def storage_delete(ctx: RunContext[Any], key: str) -> str:
    if key in storage_backend:
        del storage_backend[key]
        return f"Deleted key: {key}"
    return f"Key not found: {key}"

# Create routing configuration
storage_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'read': ('storage_read', lambda x: {'key': x.key}),
        'write': ('storage_write', lambda x: {'key': x.key, 'data': x.data}),
        'list': ('storage_list', lambda x: {}),
        'delete': ('storage_delete', lambda x: {'key': x.key}),
    }
)

# Create the storage agent
storage = create_agentool(
    name='storage',
    input_schema=StorageOperationInput,
    routing_config=storage_routing,
    tools=[storage_read, storage_write, storage_list, storage_delete],
    description="A complete storage system with CRUD operations"
)

# Use the storage
await storage.run('{"operation": "write", "key": "user:123", "data": {"name": "Alice"}}')
result = await storage.run('{"operation": "read", "key": "user:123"}')
print(result.output)  # {"name": "Alice"}
```

### Error Handling

```python
# AgenTool handles errors gracefully
result = await storage.run('{"operation": "invalid"}')
print(result.output)  
# "Error creating input model: Input should be 'read', 'write', 'list' or 'delete'"

# Missing required fields
result = await storage.run('{"operation": "write"}')
print(result.output)
# "Error creating input model: 'key' is required for write operation"
```

### Integration with LLM Agents

```python
from pydantic_ai import Agent

# Create a hybrid system
llm_agent = Agent('openai:gpt-4', system_prompt="You are a helpful assistant")
storage_agent = create_agentool(...)  # As above

# Use LLM for complex queries
complex_result = await llm_agent.run("Explain quantum computing")

# Use AgenTool for structured operations  
storage_result = await storage_agent.run('{"operation": "write", "key": "quantum", "data": "..."}')
```

## Testing

AgenTools are highly testable due to their deterministic nature:

```python
import pytest
from your_module import calculator  # Your AgenTool

@pytest.mark.asyncio
async def test_calculator_add():
    result = await calculator.run('{"operation": "add", "a": 5, "b": 3}')
    assert float(result.output) == 8.0

@pytest.mark.asyncio
async def test_calculator_validation():
    # Invalid operation
    result = await calculator.run('{"operation": "invalid", "a": 1, "b": 2}')
    assert "Error creating input model" in result.output
```

## Best Practices

1. **Schema Design**: Keep input schemas focused and well-documented
2. **Error Messages**: Return clear error messages for debugging
3. **Tool Functions**: Keep tools simple and focused on one task
4. **Routing Logic**: Use descriptive operation names
5. **Testing**: Write comprehensive tests for all operations
6. **Documentation**: Document schemas and operations clearly

## Contributing

AgenTool is part of the pydantic-ai ecosystem. Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

AgenTool is released under the same license as pydantic-ai (MIT License).

## Acknowledgments

Built on top of the excellent [pydantic-ai](https://github.com/pydantic/pydantic-ai) framework by the Pydantic team.