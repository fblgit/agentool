# AgenTool Dependency Injection Documentation

## Overview

The AgenTool framework provides a clean, automatic dependency injection system that simplifies building multi-agent systems. This documentation covers the design, implementation, and usage of the dependency injection pattern in AgenTool.

## Key Concepts

### 1. Automatic Registration

When you create an AgenTool using `create_agentool()`, it is automatically registered with the global injector:

```python
# This automatically registers the agent with the injector
agent = create_agentool(
    name='kv_storage',
    input_schema=KVStorageInput,
    routing_config=routing,
    tools=[kv_get, kv_set],
    system_prompt="Key-value storage operations"
)
```

No manual registration is needed - the factory handles it for you.

### 2. The Global Injector

The `AgenToolInjector` is a singleton that manages all agent instances and their dependencies:

```python
from src.agentool import get_injector

# Get the global injector instance
injector = get_injector()

# Run an agent with automatic dependency injection
result = await injector.run('kv_storage', json.dumps({
    "operation": "set",
    "key": "foo",
    "value": "bar"
}))
```

### 3. Dependency Declaration

Dependencies are declared when creating an AgenTool:

```python
session_agent = create_agentool(
    name='session',
    input_schema=SessionInput,
    routing_config=session_routing,
    tools=[session_create, session_get],
    dependencies=["kv_storage"],  # Declares dependency on kv_storage
    system_prompt="Session management using KV storage"
)
```

### 4. Using Dependencies in Tools

Tools access other agents through the injector:

```python
async def session_create(ctx: RunContext[None], user_id: str, data: dict) -> dict:
    """Create a new session using KV storage."""
    
    # Get the injector
    injector = get_injector()
    
    # Call the KV storage agent
    kv_input = {
        "operation": "set",
        "key": f"session:{user_id}",
        "value": data
    }
    result = await injector.run('kv_storage', json.dumps(kv_input))
    
    # Parse the response
    kv_output = json.loads(result.output)
    
    return {"success": True, "session_id": f"session:{user_id}"}
```

## Architecture Pattern

### Layer-Based Architecture

The dependency injection system supports building layered architectures:

```
Layer 3: HTTP Client
    ↓ (depends on)
Layer 2: Session Manager  
    ↓ (depends on)
Layer 1: KV Storage (base primitive)
```

Each layer can only depend on layers below it, preventing circular dependencies.

### Example: Multi-Layer System

```python
# Layer 1: Base storage primitive
kv_agent = create_agentool(
    name='kv_storage',
    input_schema=KVStorageInput,
    routing_config=kv_routing,
    tools=[kv_get, kv_set, kv_delete],
    system_prompt="Key-value storage operations"
)

# Layer 2: Session management (uses KV storage)
session_agent = create_agentool(
    name='session',
    input_schema=SessionInput,
    routing_config=session_routing,
    tools=[session_create, session_get, session_update],
    dependencies=["kv_storage"],
    system_prompt="Session management using KV storage"
)

# Layer 3: HTTP client (uses sessions)
http_agent = create_agentool(
    name='http',
    input_schema=HttpInput,
    routing_config=http_routing,
    tools=[http_request, http_get, http_post],
    dependencies=["session"],
    system_prompt="HTTP client with session support"
)
```

## JSON String Communication

AgenTools communicate using JSON strings to maintain consistency with pydantic-ai's design:

```python
# Input is always a JSON string
input_data = json.dumps({
    "operation": "get",
    "key": "user:123"
})

# Run the agent
result = await injector.run('kv_storage', input_data)

# Output is a JSON string in result.output
output_data = json.loads(result.output)
```

## Testing with Dependency Override

The injector supports dependency overrides for testing:

```python
# Create a mock KV storage for testing
mock_kv = create_agentool(
    name='mock_kv_storage',
    input_schema=KVStorageInput,
    routing_config=mock_routing,
    tools=[mock_kv_get, mock_kv_set],
    system_prompt="Mock KV storage for testing"
)

# Override the real KV storage with the mock
with injector.override(kv_storage=mock_kv):
    # This will use the mock KV storage
    result = await injector.run('session', json.dumps({
        "operation": "create",
        "user_id": "test_user"
    }))
```

## Benefits

1. **Automatic Wiring**: No manual dependency injection needed
2. **Type Safety**: Full type checking with pydantic models
3. **Testability**: Easy to mock dependencies for testing
4. **Modularity**: Each agent is self-contained with clear interfaces
5. **Observability**: Integrates with Logfire for tracing multi-agent calls

## Best Practices

### 1. Use Clear Naming Conventions

```python
# Good: Clear, descriptive names
create_agentool(name='user_session_manager', ...)
create_agentool(name='redis_cache', ...)

# Bad: Ambiguous names
create_agentool(name='manager', ...)
create_agentool(name='cache', ...)
```

### 2. Define Clear Input/Output Schemas

```python
class StorageInput(BaseModel):
    """Clear schema with descriptions."""
    operation: Literal['get', 'set', 'delete'] = Field(
        description="The storage operation to perform"
    )
    key: str = Field(description="The storage key")
    value: Optional[Any] = Field(None, description="Value for set operation")
```

### 3. Handle Errors Gracefully

```python
async def my_tool(ctx: RunContext[None], key: str) -> dict:
    """Tool with proper error handling."""
    try:
        injector = get_injector()
        result = await injector.run('dependency', json.dumps({"key": key}))
        output = json.loads(result.output)
        
        if not output.get("success"):
            return {"success": False, "error": "Dependency failed"}
            
        return {"success": True, "data": output["data"]}
        
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 4. Document Dependencies

Always list dependencies explicitly:

```python
create_agentool(
    name='payment_processor',
    dependencies=["user_session", "payment_gateway", "audit_log"],
    description="Payment processor with session validation and audit logging",
    # ... other parameters
)
```

## Implementation Details

### The Injector Class

The `AgenToolInjector` manages agent instances:

```python
class AgenToolInjector:
    def register(self, name: str, agent: Agent) -> None:
        """Register an agent with the injector."""
        
    def get(self, name: str) -> Agent:
        """Get an agent by name."""
        
    async def run(self, agent_name: str, input_data: str, **kwargs) -> Any:
        """Run an agent with automatic dependency injection."""
        
    @contextmanager
    def override(self, **overrides: Agent):
        """Temporarily override dependencies for testing."""
```

### Automatic Registration in Factory

The `create_agentool()` factory automatically registers agents:

```python
def create_agentool(...) -> Agent:
    # ... create agent ...
    
    # Register with the global injector
    from .core.injector import get_injector
    injector = get_injector()
    injector.register(name, agent)
    
    return agent
```

### JSON String Consistency

The `AgenToolManager` ensures consistent JSON string output:

```python
# In manager.py
async def __call__(self, ctx: RunContext[Any], **kwargs) -> Any:
    # ... process request ...
    
    # Ensure dict results are JSON-serialized
    if isinstance(result, dict):
        return json.dumps(result)
    return result
```

## Complete Example

Here's a complete example showing all concepts:

```python
#!/usr/bin/env python3
"""Complete dependency injection example."""

import asyncio
import json
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from src.agentool import create_agentool, get_injector
from src.agentool.core.registry import RoutingConfig

# Define schemas
class CacheInput(BaseModel):
    operation: Literal['get', 'set'] = Field(description="Cache operation")
    key: str = Field(description="Cache key")
    value: Optional[str] = Field(None, description="Value for set")

class ServiceInput(BaseModel):
    operation: Literal['process'] = Field(description="Service operation")
    data: str = Field(description="Data to process")

# Cache agent (base layer)
async def cache_get(ctx: RunContext[None], key: str) -> dict:
    # Simulate cache lookup
    return {"success": True, "value": f"cached_{key}"}

async def cache_set(ctx: RunContext[None], key: str, value: str) -> dict:
    # Simulate cache store
    return {"success": True, "stored": True}

cache_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'get': ('cache_get', lambda x: {'key': x.key}),
        'set': ('cache_set', lambda x: {'key': x.key, 'value': x.value})
    }
)

cache_agent = create_agentool(
    name='cache',
    input_schema=CacheInput,
    routing_config=cache_routing,
    tools=[cache_get, cache_set]
)

# Service agent (uses cache)
async def process_data(ctx: RunContext[None], data: str) -> dict:
    injector = get_injector()
    
    # Check cache first
    cache_result = await injector.run('cache', json.dumps({
        "operation": "get",
        "key": f"result_{data}"
    }))
    
    cache_output = json.loads(cache_result.output)
    if cache_output.get("value"):
        return {"success": True, "result": cache_output["value"], "from_cache": True}
    
    # Process data
    result = f"processed_{data}"
    
    # Store in cache
    await injector.run('cache', json.dumps({
        "operation": "set",
        "key": f"result_{data}",
        "value": result
    }))
    
    return {"success": True, "result": result, "from_cache": False}

service_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'process': ('process_data', lambda x: {'data': x.data})
    }
)

service_agent = create_agentool(
    name='service',
    input_schema=ServiceInput,
    routing_config=service_routing,
    tools=[process_data],
    dependencies=["cache"]
)

# Use the agents
async def main():
    injector = get_injector()
    
    # First call - will process and cache
    result1 = await injector.run('service', json.dumps({
        "operation": "process",
        "data": "hello"
    }))
    print(f"First call: {result1.output}")
    
    # Second call - will use cache
    result2 = await injector.run('service', json.dumps({
        "operation": "process",
        "data": "hello"
    }))
    print(f"Second call: {result2.output}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

The AgenTool dependency injection system provides a clean, automatic way to build complex multi-agent systems. By following the patterns and best practices outlined in this documentation, you can create maintainable, testable, and observable agent architectures.