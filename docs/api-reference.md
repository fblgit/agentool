# AgenTool API Reference

This document provides comprehensive API documentation for all modules, classes, and functions in the AgenTool framework.

## Table of Contents

- [Core Module (`agentool`)](#core-module-agentool)
- [Base Schemas (`agentool.base`)](#base-schemas-agentoolbase)
- [Factory Functions (`agentool.factory`)](#factory-functions-agenttoolfactory)
- [Core Components (`agentool.core`)](#core-components-agentoolcore)
  - [Model (`agentool.core.model`)](#model-agentoolcoremodel)
  - [Manager (`agentool.core.manager`)](#manager-agentoolcoremanager)
  - [Registry (`agentool.core.registry`)](#registry-agentoolcoreregistry)
  - [Injector (`agentool.core.injector`)](#injector-agentoolcoreinjector)

---

## Core Module (`agentool`)

The main module that provides the primary exports and initialization functionality.

### Exports

```python
from agentool import (
    # Core components
    AgenToolModel,
    AgenToolManager, 
    AgenToolRegistry,
    AgenToolConfig,
    RoutingConfig,
    
    # Dependency injection
    get_injector,
    InjectedDeps,
    
    # Base schemas and factory
    BaseOperationInput,
    create_agentool,
    register_agentool_models,
)
```

### Functions

#### `setup() -> None`

Sets up the AgenTool framework by registering model providers.

**Usage:**
```python
from agentool import setup
setup()  # Called automatically on import
```

**Note:** This function is called automatically when importing agentool, so manual invocation is typically not necessary.

---

## Base Schemas (`agentool.base`)

Provides base schema classes for operation-based AgenTools.

### Classes

#### `BaseOperationInput`

Base schema for operation-based AgenTools that use an operation field to route to different tools.

**Inheritance:** `pydantic.BaseModel`

**Attributes:**
- `operation: str` - The operation to perform (typically overridden with Literal types)

**Configuration:**
- `model_config = ConfigDict(extra='forbid')` - Strict validation by default

**Example:**
```python
from typing import Literal, Optional
from agentool import BaseOperationInput

class StorageInput(BaseOperationInput):
    operation: Literal['read', 'write', 'delete']
    key: str
    value: Optional[str] = None
```

---

## Factory Functions (`agentool.factory`)

Provides factory functions for creating and registering AgenTool instances.

### Functions

#### `extract_tool_metadata(tool_func: Callable) -> ToolMetadata`

Extracts metadata from a tool function for registration and documentation purposes.

**Parameters:**
- `tool_func: Callable` - The tool function to analyze

**Returns:**
- `ToolMetadata` - Object containing extracted metadata

**Extracted Information:**
- Function name
- Docstring (first line used as description)
- Async/sync status
- Parameter names and types
- Return type annotation

**Example:**
```python
async def add_numbers(ctx: RunContext[Any], a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

metadata = extract_tool_metadata(add_numbers)
# metadata.name == 'add_numbers'
# metadata.description == 'Add two numbers together.'
# metadata.is_async == True
# metadata.parameters == ['a', 'b']
```

#### `infer_output_type(tools_metadata: List[ToolMetadata]) -> Optional[Type[Any]]`

Infers the output type from tool return annotations for automatic validation.

**Parameters:**
- `tools_metadata: List[ToolMetadata]` - Metadata from all tools

**Returns:**
- `Optional[Type[Any]]` - The inferred common output type, or None if incompatible

**Inference Rules:**
1. Returns common BaseModel type if all tools return the same type
2. Returns BaseModel type if some tools return dict and others return that BaseModel
3. Returns None for incompatible types or built-in types (str, int, etc.)
4. Skips generic types (e.g., Dict[str, Any])

**Example:**
```python
# All tools return the same BaseModel
class Result(BaseModel):
    value: str

async def tool1(ctx) -> Result: ...
async def tool2(ctx) -> Result: ...

# infer_output_type will return Result

# Mixed dict/BaseModel (compatible)
async def tool3(ctx) -> dict: ...  # Can be converted to Result
async def tool4(ctx) -> Result: ...

# infer_output_type will return Result
```

#### `create_agentool(...) -> Agent`

Main factory function to create a configured AgenTool agent.

**Signature:**
```python
def create_agentool(
    name: str,
    input_schema: Type[BaseModel],
    routing_config: RoutingConfig,
    tools: List[Callable],
    system_prompt: Optional[str] = None,
    description: Optional[str] = None,
    deps_type: Optional[Type[Any]] = None,
    output_type: Optional[Type[Any]] = None,
    version: str = "1.0.0",
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[str]] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    **agent_kwargs,
) -> Agent
```

**Parameters:**
- `name: str` - Unique identifier for the AgenTool
- `input_schema: Type[BaseModel]` - Pydantic model for input validation
- `routing_config: RoutingConfig` - Configuration for operation routing
- `tools: List[Callable]` - List of tool functions to register
- `system_prompt: Optional[str]` - Custom system prompt (default: auto-generated)
- `description: Optional[str]` - Description of the AgenTool
- `deps_type: Optional[Type[Any]]` - Type for dependency injection
- `output_type: Optional[Type[Any]]` - Expected output type (auto-inferred if None)
- `version: str` - Version string (default: "1.0.0")
- `tags: Optional[List[str]]` - Tags for categorization
- `dependencies: Optional[List[str]]` - List of required dependencies
- `examples: Optional[List[Dict[str, Any]]]` - Usage examples
- `**agent_kwargs` - Additional arguments passed to Agent constructor

**Returns:**
- `Agent` - Configured pydantic-ai Agent ready to use

**Process:**
1. Extracts metadata from all tools
2. Infers output type if not provided
3. Creates and registers configuration in registry
4. Creates Agent with AgenToolModel
5. Registers manager and actual tools
6. Registers with dependency injector
7. Returns configured agent

**Example:**
```python
from agentool import create_agentool, RoutingConfig
from pydantic import BaseModel

class CalculatorInput(BaseModel):
    operation: str
    a: float
    b: float

async def add(ctx, a: float, b: float) -> float:
    return a + b

async def multiply(ctx, a: float, b: float) -> float:
    return a * b

routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'add': ('add', lambda x: {'a': x.a, 'b': x.b}),
        'multiply': ('multiply', lambda x: {'a': x.a, 'b': x.b}),
    }
)

calculator = create_agentool(
    name='calculator',
    input_schema=CalculatorInput,
    routing_config=routing,
    tools=[add, multiply],
    description="Mathematical calculator",
    tags=['math', 'calculator'],
    version='1.0.0'
)
```

#### `register_agentool_models() -> None`

Registers the AgenTool model provider with pydantic-ai's model inference system.

**Process:**
1. Patches `pydantic_ai.models.infer_model`
2. Adds support for 'agentool:*' model strings
3. Creates `AgenToolModel` instances for matching patterns

**Usage:**
```python
from agentool import register_agentool_models
register_agentool_models()  # Usually called automatically

# Now you can use agentool model strings
from pydantic_ai import Agent
agent = Agent('agentool:storage')  # Creates AgenToolModel('storage')
```

**Note:** This function is called automatically when importing agentool.

---

## Core Components (`agentool.core`)

The core components that implement the AgenTool framework functionality.

### Model (`agentool.core.model`)

#### `AgenToolModel`

A synthetic LLM model provider that converts JSON input to tool calls.

**Inheritance:** `pydantic_ai.models.Model`

**Attributes:**
- `name: str` - The name of the AgenTool
- `_model_name: str` - Full model name (e.g., 'agentool:storage')
- `_system: str` - System identifier ('agentool')

**Constructor:**
```python
def __init__(self, name: str, **kwargs)
```

**Methods:**

##### `async def request(...) -> ModelResponse`

Processes a request and generates appropriate tool calls or responses.

**Signature:**
```python
async def request(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse
```

**Process:**
1. **Phase 1 (Initial):** Extracts JSON from user message, generates tool call to `__agentool_manager__`
2. **Phase 2 (Tool Response):** Processes tool results and returns final text response

**Error Handling:**
- Returns error for missing configuration
- Returns error for invalid JSON input
- Handles tool execution errors gracefully

##### `async def request_stream(...) -> AsyncIterator[StreamedResponse]`

Not supported for AgenTools (raises `NotImplementedError`).

##### Properties

- `model_name: str` - Returns the full model name
- `system: str` - Returns the system identifier

**Example:**
```python
from agentool.core import AgenToolModel

model = AgenToolModel('calculator')
# model.model_name == 'agentool:calculator'
# model.system == 'agentool'
```

### Manager (`agentool.core.manager`)

#### `AgenToolManager`

Handles routing and payload transformation for AgenTools.

**Constructor:**
```python
def __init__(
    self, 
    name: str, 
    config: AgenToolConfig, 
    tool_functions: Dict[str, Callable] = None
)
```

**Attributes:**
- `name: str` - The AgenTool name
- `config: AgenToolConfig` - Configuration object
- `input_schema: Type[BaseModel]` - Input validation schema
- `routing_config: RoutingConfig` - Routing configuration
- `tool_functions: Dict[str, Callable]` - Tool name to function mapping

**Methods:**

##### `async def __call__(ctx: RunContext[Any], **kwargs) -> Any`

Main entry point when called as a tool. Handles the complete routing process.

**Process:**
1. Creates input model instance from kwargs
2. Extracts operation field value
3. Looks up routing configuration
4. Transforms input data for target tool
5. Executes tool function
6. Processes output based on output_type configuration

**Error Handling:**
- Validates input against schema
- Checks for unknown operations
- Handles tool execution errors
- Validates output types

**Example:**
```python
# Called automatically by pydantic-ai
result = await manager(ctx, operation='add', a=5, b=3)
```

##### `def get_tool_schema() -> Dict[str, Any]`

Returns the JSON schema for input validation.

**Returns:**
- `Dict[str, Any]` - JSON schema derived from input_schema

**Usage:**
```python
schema = manager.get_tool_schema()
# Returns schema with properties, required fields, etc.
```

### Registry (`agentool.core.registry`)

#### Data Classes

##### `RoutingConfig`

Configuration for routing operations to tools.

**Attributes:**
- `operation_field: str = 'operation'` - Field name containing the operation
- `operation_map: Dict[str, Tuple[str, Callable]]` - Maps operation values to (tool_name, transform_func)

**Example:**
```python
routing = RoutingConfig(
    operation_field='action',  # Custom field name
    operation_map={
        'create': ('create_item', lambda x: {'data': x.payload}),
        'update': ('update_item', lambda x: {'id': x.id, 'data': x.payload}),
        'delete': ('delete_item', lambda x: {'id': x.id}),
    }
)
```

##### `ToolMetadata`

Metadata about a tool function.

**Attributes:**
- `name: str` - Tool function name
- `description: Optional[str]` - Description from docstring
- `is_async: bool` - Whether the tool is async
- `parameters: List[str]` - Parameter names
- `parameter_types: Dict[str, str]` - Parameter types as strings
- `return_type: Optional[str]` - Return type as string
- `return_type_annotation: Optional[Type[Any]]` - Actual return type object

##### `AgenToolConfig`

Complete configuration for an AgenTool.

**Attributes:**
- `input_schema: Type[BaseModel]` - Pydantic model for input validation
- `routing_config: RoutingConfig` - Operation routing configuration
- `output_type: Optional[Type[BaseModel]]` - Expected output type
- `description: Optional[str]` - Human-readable description
- `version: str` - Version string
- `tags: List[str]` - Categorization tags
- `tools_metadata: List[ToolMetadata]` - Tool function metadata
- `dependencies: List[str]` - Required dependencies
- `examples: List[Dict[str, Any]]` - Usage examples
- `created_at: datetime` - Creation timestamp
- `updated_at: datetime` - Last update timestamp

#### `AgenToolRegistry`

Global registry for AgenTool configurations.

**Class Methods:**

##### `register(name: str, config: AgenToolConfig) -> None`

Registers an AgenTool configuration.

**Example:**
```python
AgenToolRegistry.register('storage', config)
```

##### `get(name: str) -> Optional[AgenToolConfig]`

Retrieves an AgenTool configuration by name.

**Example:**
```python
config = AgenToolRegistry.get('storage')
if config:
    print(config.description)
```

##### `list_names() -> List[str]`

Lists all registered AgenTool names.

**Example:**
```python
names = AgenToolRegistry.list_names()
# ['storage', 'calculator', 'weather']
```

##### `clear() -> None`

Clears all registered configurations (useful for testing).

##### `list_detailed() -> List[Dict[str, Any]]`

Returns detailed information about all registered AgenTools.

**Returns:**
```python
[
    {
        "name": "calculator",
        "version": "1.0.0",
        "description": "Mathematical calculator",
        "tags": ["math"],
        "operations": ["add", "subtract", "multiply", "divide"],
        "tools": [
            {
                "name": "add",
                "async": True,
                "params": ["a", "b"],
                "description": "Add two numbers"
            }
        ],
        "input_schema": {...},
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00"
    }
]
```

##### `search(tags: Optional[List[str]] = None, name_pattern: Optional[str] = None) -> List[str]`

Searches AgenTools by tags or name pattern.

**Example:**
```python
# Find all math-related tools
math_tools = AgenToolRegistry.search(tags=['math'])

# Find tools with 'calc' in the name
calc_tools = AgenToolRegistry.search(name_pattern='calc')
```

##### `get_schema(name: str) -> Optional[Dict[str, Any]]`

Gets the JSON schema for an AgenTool's input.

##### `get_tools_info(name: str) -> Optional[List[Dict[str, Any]]]`

Gets information about tools used by an AgenTool.

##### `get_operations(name: str) -> Optional[Dict[str, Dict[str, Any]]]`

Gets available operations for an AgenTool.

**Returns:**
```python
{
    "add": {
        "tool": "add_numbers",
        "description": "Add two numbers together",
        "parameters": ["a", "b"]
    },
    "subtract": {
        "tool": "subtract_numbers", 
        "description": "Subtract two numbers",
        "parameters": ["a", "b"]
    }
}
```

##### `export_catalog() -> Dict[str, Any]`

Exports the complete AgenTool catalog.

**Returns:**
```python
{
    "version": "1.0.0",
    "generated_at": "2024-01-01T00:00:00",
    "total_agentools": 3,
    "agentools": [...]  # Detailed list
}
```

##### `generate_markdown_docs() -> str`

Generates markdown documentation for all registered AgenTools.

**Returns:** Complete markdown documentation string with:
- AgenTool descriptions
- Operations and tools
- Input schemas
- Examples

##### `generate_dependency_graph(include_tools: bool = True) -> Dict[str, Dict[str, List[str]]]`

Generates dependency graphs showing relationships.

**Returns:**
```python
{
    "agentools": {
        "agent_a": ["agent_b", "agent_c"],  # AgenTool dependencies
        "agent_b": [],
        "agent_c": []
    },
    "tools": {
        "agent_a": ["tool1", "tool2"],  # Tool dependencies
        "agent_b": ["tool3"],
        "agent_c": ["tool4", "tool5"]
    }
}
```

##### `generate_api_spec() -> Dict[str, Any]`

Generates OpenAPI-like specification for all AgenTools.

**Returns:** OpenAPI 3.0 compatible specification with endpoints for each AgenTool and operation.

### Injector (`agentool.core.injector`)

#### Utility Functions

##### `serialize_to_json_string(data: Any) -> str`

Serializes various Python types to JSON strings.

**Supported Types:**
- Strings (validates existing JSON)
- Dicts, lists, tuples
- BaseModel instances (uses `model_dump_json()`)
- Basic types (int, float, bool, None)
- Datetime objects (ISO format)
- Decimal objects

**Example:**
```python
from datetime import datetime
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Different input types
serialize_to_json_string({"key": "value"})  # '{"key": "value"}'
serialize_to_json_string(User(name="Alice", age=30))  # '{"name":"Alice","age":30}'
serialize_to_json_string(datetime.now())  # '"2024-01-01T00:00:00"'
serialize_to_json_string("hello")  # '"hello"'
```

##### `validate_json_string(data: str) -> bool`

Validates if a string is valid JSON.

**Example:**
```python
validate_json_string('{"valid": true}')  # True
validate_json_string('invalid json')     # False
```

#### `AgenToolInjector`

Manages dependency injection for AgenTools.

**Constructor:**
```python
def __init__(self)
```

**Attributes:**
- `_instances: Dict[str, Agent]` - Cached agent instances
- `_dependency_overrides: Dict[str, Agent]` - Temporary overrides

**Methods:**

##### `register(name: str, agent: Agent) -> None`

Registers an agent instance.

**Example:**
```python
injector.register('storage', storage_agent)
```

##### `get(name: str) -> Agent`

Retrieves an agent instance by name with dependency resolution.

**Process:**
1. Checks for dependency overrides
2. Returns cached instance if available
3. Validates configuration exists
4. Raises error if instance not found

##### `create_deps(agent_name: str) -> Optional[Any]`

Creates a dependencies object based on agent configuration.

**Returns:** Dynamic object with dependencies as attributes, or None if no deps needed.

**Example:**
```python
# If agent has dependencies: ['storage', 'cache']
deps = injector.create_deps('complex_agent')
# deps.storage = storage_agent
# deps.cache = cache_agent
```

##### `async def run(agent_name: str, input_data: Union[str, Dict, BaseModel, Any], **kwargs) -> Any`

Runs an agent with automatic dependency injection and serialization.

**Parameters:**
- `agent_name: str` - Name of agent to run
- `input_data: Union[str, Dict, BaseModel, Any]` - Input data (auto-serialized)
- `**kwargs` - Additional arguments for agent

**Example:**
```python
# With different input types
result = await injector.run('storage', {"operation": "get", "key": "foo"})
result = await injector.run('storage', StorageInput(operation="get", key="foo"))
result = await injector.run('storage', '{"operation": "get", "key": "foo"}')
```

##### `@contextmanager override(**overrides: Agent)`

Context manager for temporary dependency overrides.

**Example:**
```python
mock_storage = create_mock_storage_agent()
with injector.override(storage=mock_storage):
    result = await injector.run('complex_agent', input_data)
    # Uses mock_storage instead of real storage
```

##### `clear()`

Clears all cached instances and overrides.

#### `InjectedDeps[T]`

Generic base class for dependency injection with type hints.

**Attributes:**
- `_injector: AgenToolInjector` - Injector instance (auto-created)

**Methods:**

##### `get_agent(name: str) -> Agent`

Gets an agent from the injector.

##### `async def call_agent(name: str, input_data: Union[str, Dict, BaseModel, Any]) -> Any`

Calls an agent with automatic dependency injection.

**Example:**
```python
@dataclass
class MyDeps(InjectedDeps):
    async def process_data(self, data: str) -> str:
        # Call storage agent
        result = await self.call_agent('storage', {
            "operation": "process", 
            "data": data
        })
        return result.output
```

#### Global Functions

##### `get_injector() -> AgenToolInjector`

Returns the global injector instance.

**Example:**
```python
from agentool.core.injector import get_injector

injector = get_injector()
await injector.run('storage', input_data)
```

---

## Usage Patterns

### 1. Basic AgenTool Creation

```python
from agentool import create_agentool, RoutingConfig, BaseOperationInput
from typing import Literal

class MyInput(BaseOperationInput):
    operation: Literal['read', 'write']
    key: str
    value: Optional[str] = None

async def read_tool(ctx, key: str) -> str:
    return f"Reading {key}"

async def write_tool(ctx, key: str, value: str) -> str:
    return f"Writing {value} to {key}"

routing = RoutingConfig(
    operation_map={
        'read': ('read_tool', lambda x: {'key': x.key}),
        'write': ('write_tool', lambda x: {'key': x.key, 'value': x.value}),
    }
)

agent = create_agentool(
    name='simple_storage',
    input_schema=MyInput,
    routing_config=routing,
    tools=[read_tool, write_tool]
)
```

### 2. Advanced Configuration

```python
from pydantic import BaseModel

class ComplexOutput(BaseModel):
    result: str
    metadata: Dict[str, Any]
    timestamp: datetime

class ComplexInput(BaseOperationInput):
    operation: Literal['complex_op']
    params: Dict[str, Any]

async def complex_tool(ctx, params: Dict[str, Any]) -> ComplexOutput:
    return ComplexOutput(
        result="processed",
        metadata=params,
        timestamp=datetime.now()
    )

agent = create_agentool(
    name='complex_agent',
    input_schema=ComplexInput,
    routing_config=routing,
    tools=[complex_tool],
    output_type=ComplexOutput,  # Enables type validation
    description="Complex processing agent",
    tags=['advanced', 'processing'],
    dependencies=['storage', 'cache'],
    examples=[
        {
            "input": {"operation": "complex_op", "params": {"key": "value"}},
            "output": {"result": "processed", "metadata": {"key": "value"}}
        }
    ]
)
```

### 3. Dependency Injection

```python
from agentool.core.injector import get_injector

# Create dependent agents
storage_agent = create_agentool(...)
cache_agent = create_agentool(...)

# Use injector for complex workflows
injector = get_injector()

async def complex_workflow():
    # Agents are automatically wired with their dependencies
    result1 = await injector.run('storage', storage_input)
    result2 = await injector.run('cache', cache_input)
    
    # Override for testing
    with injector.override(storage=mock_storage):
        test_result = await injector.run('complex_agent', test_input)
```

This API reference provides complete documentation for all public interfaces in the AgenTool framework, enabling developers to effectively use and extend the system.