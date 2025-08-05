# AgenTool Framework Documentation

**Version**: 1.0.0  
**Framework**: A deterministic tool execution framework for pydantic-ai

## Overview

AgenTool is a framework that provides deterministic, schema-driven tool execution within [pydantic-ai](https://github.com/pydantic/pydantic-ai). It acts as a synthetic LLM model provider that routes JSON input to appropriate tools based on configuration, offering a cost-effective and predictable alternative to LLM calls for structured operations.

The framework bridges the gap between LLM-powered agents and programmatic tool execution, enabling developers to create hybrid systems that combine the intelligence of LLMs with the reliability of deterministic code execution.

## Key Features

- **🎯 Schema-Driven Validation**: Uses Pydantic models for input validation and type safety
- **🔀 Intelligent Routing**: Routes operations to tools based on configurable operation fields
- **🔧 Tool Integration**: Works seamlessly with any async Python function as a tool
- **📦 Zero Additional Dependencies**: Built on top of pydantic-ai's existing infrastructure
- **🎭 LLM Interface Compatibility**: Maintains the same interface as LLM-based agents
- **💰 Cost Effective**: No LLM API calls for deterministic operations
- **🧪 Highly Testable**: Predictable behavior makes testing straightforward
- **🔌 Dependency Injection**: Built-in system for inter-agent communication
- **📊 Rich Metadata**: Comprehensive tool and schema introspection capabilities

## When to Use AgenTool

### Perfect Use Cases

- **Structured Operations**: CRUD operations, API calls, data transformations
- **Cost Optimization**: Avoiding LLM API costs for simple routing logic
- **Testing Scenarios**: Predictable behavior for unit and integration tests
- **Hybrid Systems**: Mixing LLM intelligence with deterministic execution
- **Multi-Agent Systems**: Complex agent hierarchies with deterministic sub-agents
- **API Gateway Patterns**: Routing requests to different handlers
- **Workflow Orchestration**: Step-by-step process execution

### When NOT to Use AgenTool

- **Natural Language Processing**: Tasks requiring language understanding
- **Creative Content Generation**: Text/image/code generation tasks
- **Complex Reasoning**: Multi-step reasoning or inference
- **Dynamic Decision Making**: Scenarios requiring contextual judgment

## Quick Start

```python
from agentool import create_agentool, RoutingConfig
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from typing import Any, Literal

# 1. Define your input schema
class CalculatorInput(BaseModel):
    operation: Literal['add', 'subtract', 'multiply', 'divide']
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

# 2. Create your tool functions
async def add(ctx: RunContext[Any], a: float, b: float) -> float:
    return a + b

async def subtract(ctx: RunContext[Any], a: float, b: float) -> float:
    return a - b

async def multiply(ctx: RunContext[Any], a: float, b: float) -> float:
    return a * b

async def divide(ctx: RunContext[Any], a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# 3. Configure routing
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'add': ('add', lambda x: {'a': x.a, 'b': x.b}),
        'subtract': ('subtract', lambda x: {'a': x.a, 'b': x.b}),
        'multiply': ('multiply', lambda x: {'a': x.a, 'b': x.b}),
        'divide': ('divide', lambda x: {'a': x.a, 'b': x.b})
    }
)

# 4. Create the AgenTool
calculator = create_agentool(
    name='calculator',
    input_schema=CalculatorInput,
    routing_config=routing,
    tools=[add, subtract, multiply, divide],
    description="A mathematical calculator with basic operations"
)

# 5. Use it like any pydantic-ai agent
async def main():
    result = await calculator.run('{"operation": "add", "a": 5, "b": 3}')
    print(result.output)  # "8.0"
    
    result = await calculator.run('{"operation": "multiply", "a": 4, "b": 7}')
    print(result.output)  # "28.0"

# Run the example
import asyncio
asyncio.run(main())
```

## Architecture Overview

The AgenTool framework consists of five main components:

### 1. **AgenToolModel** (`core/model.py`)
A synthetic LLM model provider that:
- Parses JSON input from user messages
- Generates tool calls to the routing manager
- Returns tool results as model responses
- Integrates seamlessly with pydantic-ai's model system

### 2. **AgenToolManager** (`core/manager.py`)
The routing engine that:
- Validates input against schemas
- Routes operations to appropriate tools
- Transforms data between input schema and tool signatures
- Executes tools and handles results

### 3. **AgenToolRegistry** (`core/registry.py`)
Global configuration storage that:
- Stores input schemas and routing configurations
- Enables dynamic tool lookup and introspection
- Supports multiple AgenTool instances
- Provides rich metadata and documentation generation

### 4. **AgenToolInjector** (`core/injector.py`)
Dependency injection system that:
- Manages inter-agent dependencies
- Provides automatic JSON serialization
- Supports dependency overrides for testing
- Enables complex multi-agent architectures

### 5. **Factory Functions** (`factory.py`)
High-level creation utilities that:
- Simplify AgenTool creation and registration
- Extract metadata from tool functions
- Handle automatic output type inference
- Register model providers with pydantic-ai

## Project Structure

```
src/agentool/
├── __init__.py              # Main exports and auto-setup
├── base.py                  # Base schemas for operation-based tools
├── factory.py               # Factory functions for creating AgenTools
└── core/                    # Core framework components
    ├── __init__.py          # Core exports
    ├── model.py             # Synthetic LLM model provider
    ├── manager.py           # Routing and execution manager
    ├── registry.py          # Global configuration registry
    └── injector.py          # Dependency injection system
```

## Supported Input/Output Types

### Input Types
- **JSON Strings**: `'{"operation": "add", "a": 5, "b": 3}'`
- **Python Dictionaries**: `{"operation": "add", "a": 5, "b": 3}`
- **Pydantic Models**: `CalculatorInput(operation="add", a=5, b=3)`
- **Basic Types**: Automatically serialized to JSON

### Output Types
- **Automatic Inference**: Framework infers output types from tool return annotations
- **Pydantic Models**: Structured output with validation
- **JSON Serializable Types**: Dicts, lists, basic types
- **Error Handling**: Graceful error responses with meaningful messages

## Error Handling

AgenTool provides comprehensive error handling:

```python
# Schema validation errors
result = await calculator.run('{"operation": "invalid", "a": 1, "b": 2}')
print(result.output)  
# "Unknown operation 'invalid'. Available operations: ['add', 'subtract', 'multiply', 'divide']"

# Missing required fields
result = await calculator.run('{"operation": "add", "a": 5}')
print(result.output)
# "Error creating input model: Field required [type=missing, input={...}]"

# Tool execution errors
result = await calculator.run('{"operation": "divide", "a": 10, "b": 0}')
print(result.output)
# "Error calling tool 'divide': Cannot divide by zero"
```

## Integration with pydantic-ai

AgenTool integrates seamlessly with pydantic-ai:

```python
from pydantic_ai import Agent
from agentool import AgenToolModel

# Direct model usage
model = AgenToolModel('calculator')
agent = Agent(model=model)

# String-based model specification (after registration)
agent = Agent('agentool:calculator')

# Mixed agent hierarchies
llm_agent = Agent('openai:gpt-4')
storage_agent = Agent('agentool:storage')

# Use LLM for complex queries
complex_result = await llm_agent.run("Explain quantum computing")

# Use AgenTool for structured operations
storage_result = await storage_agent.run('{"operation": "write", "key": "quantum", "data": "..."}')
```

## Documentation Structure

This documentation is organized into the following sections:

- **[Architecture Guide](architecture.md)**: Detailed architecture overview and component relationships
- **[API Reference](api-reference.md)**: Complete API documentation for all modules
- **[Usage Examples](usage-examples.md)**: Comprehensive examples and usage patterns
- **[Integration Guide](integration-guide.md)**: How to integrate with existing systems
- **[Testing Guide](testing-guide.md)**: Best practices for testing AgenTool-based systems

## Next Steps

1. **[Read the Architecture Guide](architecture.md)** to understand how components work together
2. **[Explore the API Reference](api-reference.md)** for detailed API documentation
3. **[Try the Examples](usage-examples.md)** to see AgenTool in action
4. **Check the existing README.md** for additional examples and use cases

## Contributing

AgenTool is part of the pydantic-ai ecosystem. Contributions are welcome! Please see the main project's contributing guidelines.

## License

AgenTool is released under the same license as pydantic-ai (MIT License).