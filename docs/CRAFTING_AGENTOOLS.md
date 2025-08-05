# Crafting Quality AgenTools: A Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [AgenTool Architecture](#agentool-architecture)
4. [Creating Your First AgenTool](#creating-your-first-agentool)
5. [Best Practices and Patterns](#best-practices-and-patterns)
6. [Advanced Features](#advanced-features)
7. [Testing Guidelines](#testing-guidelines)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

## Introduction

AgenTools are schema-driven, deterministic agents that provide structured interfaces for specific functionalities. Unlike traditional LLM-based agents, AgenTools offer predictable behavior, type safety, and seamless integration with the pydantic-ai ecosystem.

### Key Benefits
- **Type Safety**: Full Pydantic validation for inputs and outputs
- **Deterministic**: Predictable routing based on input schemas
- **Composable**: Easy integration with other AgenTools via dependency injection
- **Observable**: Built-in Logfire integration for monitoring
- **Testable**: Schema-driven design enables comprehensive testing

## Core Concepts

### 1. Input Schema
Every AgenTool starts with a Pydantic model that defines its input structure:

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class StorageInput(BaseModel):
    operation: Literal['read', 'write', 'delete', 'list'] = Field(
        description="The storage operation to perform"
    )
    key: Optional[str] = Field(None, description="Key for read/write/delete")
    value: Optional[str] = Field(None, description="Value for write operation")
    prefix: Optional[str] = Field(None, description="Prefix for list operation")
```

### 2. Routing Configuration
Routes map operations to specific tool functions:

```python
from agentool.core.registry import RoutingConfig

routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'read': ('read_value', lambda x: {'key': x.key}),
        'write': ('write_value', lambda x: {'key': x.key, 'value': x.value}),
        'delete': ('delete_value', lambda x: {'key': x.key}),
        'list': ('list_keys', lambda x: {'prefix': x.prefix or ''})
    }
)
```

### 3. Tool Functions
Implement the actual functionality as async functions:

```python
from pydantic_ai import RunContext
from typing import Any, Dict

async def read_value(ctx: RunContext[Any], key: str) -> Dict[str, Any]:
    """Read a value from storage."""
    # Implementation here
    return {"key": key, "value": stored_value}

async def write_value(ctx: RunContext[Any], key: str, value: str) -> Dict[str, Any]:
    """Write a value to storage."""
    # Implementation here
    return {"key": key, "status": "written"}
```

### 4. Output Types (Automatic Inference)
AgenTools automatically infer output types from tool return annotations. You can optionally define structured output types for validation:

```python
class StorageOutput(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
```

**Note**: If all tools return the same BaseModel type, the output_type will be automatically inferred. You only need to specify output_type explicitly if you want to override the inference or ensure type validation.

## AgenTool Architecture

### Single File Structure
Organize your AgenTool in a single file with clear sections:

```python
"""
Storage AgenTool - Provides key-value storage operations.

This AgenTool demonstrates best practices for creating
reusable, type-safe storage functionality.
"""

# 1. Imports
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from agentool import create_agentool
from agentool.core.registry import RoutingConfig

# 2. Storage Backend (if needed)
_storage: Dict[str, Any] = {}

# 3. Input Schema
class StorageInput(BaseModel):
    """Input schema for storage operations."""
    operation: Literal['read', 'write', 'delete', 'list']
    key: Optional[str] = None
    value: Optional[Any] = None
    prefix: Optional[str] = None

# 4. Output Schema (optional but recommended)
class StorageOutput(BaseModel):
    """Structured output for storage operations."""
    success: bool
    message: str
    data: Optional[Any] = None

# 5. Tool Functions
async def read_value(ctx: RunContext[Any], key: str) -> StorageOutput:
    """Read a value from storage.
    
    Raises:
        KeyError: If the key is not found in storage
    """
    if key not in _storage:
        raise KeyError(f"Key not found: {key}")
        
    return StorageOutput(
        success=True,
        message=f"Found value for key: {key}",
        data=_storage[key]
    )

# ... other tool functions ...

# 6. Routing Configuration
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'read': ('read_value', lambda x: {'key': x.key}),
        'write': ('write_value', lambda x: {'key': x.key, 'value': x.value}),
        'delete': ('delete_value', lambda x: {'key': x.key}),
        'list': ('list_keys', lambda x: {'prefix': x.prefix or ''})
    }
)

# 7. AgenTool Creation Function
def create_storage_agent():
    """Create and return the storage AgenTool."""
    return create_agentool(
        name='storage',
        input_schema=StorageInput,
        routing_config=routing,
        tools=[read_value, write_value, delete_value, list_keys],
        output_type=StorageOutput,  # Automatically inferred if not specified
        system_prompt="Handle storage operations efficiently.",
        description="Key-value storage with CRUD operations",
        version="1.0.0",
        tags=["storage", "crud", "key-value"],
        examples=[
            {
                "input": {"operation": "write", "key": "test", "value": "hello"},
                "output": {"success": true, "message": "Value written", "data": null}
            }
        ]
    )

# 8. Export
agent = create_storage_agent()
```

## Creating Your First AgenTool

### Step 1: Define Your Domain
Identify the specific functionality your AgenTool will provide:
- What operations will it support?
- What inputs does each operation need?
- What outputs should it return?

### Step 2: Design the Schema
Create a Pydantic model that captures all possible operations:

```python
from agentool.base import BaseOperationInput

class MyToolInput(BaseOperationInput):
    operation: Literal['action1', 'action2', 'action3']
    # Common fields
    id: str = Field(description="Resource identifier")
    # Operation-specific fields  
    data: Optional[Dict[str, Any]] = Field(None, description="For action1")
    config: Optional[str] = Field(None, description="For action2")
```

### Step 3: Implement Tool Functions
Write focused, single-purpose functions:

```python
async def action1(ctx: RunContext[Any], id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform action1 on the resource.
    
    Args:
        ctx: Runtime context with dependencies
        id: Resource identifier
        data: Action-specific data
        
    Returns:
        Result dictionary with status and data
        
    Raises:
        ValueError: If required data is missing
        RuntimeError: If the action fails to process
    """
    # Validate inputs - raise exceptions for errors
    if not data:
        raise ValueError("Data is required for action1 operation")
    
    # Perform action - let exceptions propagate with context
    try:
        result = await process_action1(id, data)
    except Exception as e:
        raise RuntimeError(f"Failed to process action1 for id={id}: {e}") from e
    
    # Return successful result only
    return {
        "status": "success",
        "id": id,
        "result": result
    }
```

### Step 4: Configure Routing
Map operations to functions with parameter transformers:

```python
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'action1': ('action1', lambda x: {'id': x.id, 'data': x.data}),
        'action2': ('action2', lambda x: {'id': x.id, 'config': x.config}),
        'action3': ('action3', lambda x: {'id': x.id})
    }
)
```

### Step 5: Create the AgenTool
Assemble everything with metadata:

```python
def create_my_agent():
    return create_agentool(
        name='my_tool',
        input_schema=MyToolInput,
        routing_config=routing,
        tools=[action1, action2, action3],
        output_type=MyToolOutput,  # Optional, will be inferred
        description="Handles specific domain operations",
        version="1.0.0",
        tags=["domain", "operations"]
    )
```

## Best Practices and Patterns

### 1. Schema Design
- **Use Literal types** for operation fields to enable proper routing
- **Make fields Optional** when they're only used by specific operations
- **Add descriptions** to all fields for better documentation
- **Use consistent naming** across your AgenTool ecosystem

### 2. Tool Function Guidelines
- **Single Responsibility**: Each function should do one thing well
- **Error Handling**: Use exceptions for error propagation - raise with detailed messages instead of returning error results
- **Async by Default**: Use async functions for better concurrency
- **Type Annotations**: Always include full type hints

```python
async def good_tool_function(
    ctx: RunContext[MyDeps], 
    param1: str, 
    param2: Optional[int] = None
) -> Dict[str, Any]:
    """Clear docstring explaining the function."""
    # Validation - raise exceptions for errors
    if not param1:
        raise ValueError("param1 is required and cannot be empty")
    
    # Business logic - let exceptions propagate with context
    try:
        result = await perform_operation(param1, param2)
    except SomeSpecificError as e:
        raise RuntimeError(f"Failed to perform operation for param1={param1}: {e}") from e
    
    # Return successful result only
    return {
        "success": True,
        "data": result,
        "metadata": {"param1": param1, "param2": param2}
    }
```

### 3. Error Handling Best Practices

**IMPORTANT**: Always use exceptions for error propagation in AgenTools. The system is designed to handle exceptions properly and provide meaningful error messages to users. Returning error dictionaries breaks the exception propagation chain and makes debugging difficult.

#### ✅ Correct Error Handling
```python
async def process_data(ctx: RunContext[Any], data: str) -> Dict[str, Any]:
    """Process data with proper error handling."""
    # Input validation
    if not data or not data.strip():
        raise ValueError("Input data cannot be empty or whitespace-only")
    
    # Business logic with context-aware error handling
    try:
        result = await external_api_call(data)
    except APITimeoutError as e:
        raise RuntimeError(f"API timeout while processing data: {e}") from e
    except APIError as e:
        raise RuntimeError(f"API error during data processing: {e.message}") from e
    
    # Return only on success
    return {"processed": result, "timestamp": datetime.now().isoformat()}
```

#### ❌ Incorrect Error Handling
```python
async def process_data_bad(ctx: RunContext[Any], data: str) -> Dict[str, Any]:
    """DON'T DO THIS - breaks error propagation."""
    if not data:
        return {"error": "Data required", "success": False}  # BAD
    
    try:
        result = await external_api_call(data)
        return {"success": True, "data": result}
    except Exception as e:
        return {"error": str(e), "success": False}  # BAD
```

#### Error Types to Use
- **ValueError**: For invalid input parameters or data validation failures
- **RuntimeError**: For operational failures during execution
- **KeyError**: For missing keys or identifiers
- **TypeError**: For incorrect data types
- **Custom exceptions**: For domain-specific errors with clear context

#### Exception Chaining
Always use `raise ... from e` to preserve the original exception context:
```python
try:
    result = await risky_operation()
except OriginalError as e:
    raise RuntimeError(f"Operation failed with context: {e}") from e
```

### 4. Output Type Patterns
AgenTools automatically infer output types from tool return annotations when all tools return the same BaseModel type:

```python
# Pattern 1: Consistent BaseModel output
class MyOutput(BaseModel):
    success: bool
    data: Any

async def tool1(ctx: RunContext[Any], param: str) -> MyOutput:
    return MyOutput(success=True, data=param)

async def tool2(ctx: RunContext[Any], param: int) -> MyOutput:
    return MyOutput(success=True, data=param * 2)

# Output type will be inferred as MyOutput

# Pattern 2: Mixed dict/BaseModel (compatible)
async def tool3(ctx, param: str) -> Dict[str, Any]:
    return {"success": True, "data": param}

# Can still use MyOutput as output_type, dicts will be converted

# Pattern 3: Built-in types (no output_type needed)
async def simple_tool(ctx, param: str) -> str:
    return f"Processed: {param}"
```

### 4. Dependency Injection
Use the injector for multi-agent workflows:

```python
from agentool.core.injector import get_injector

async def complex_tool(ctx: RunContext[Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """Tool that uses other AgenTools."""
    injector = get_injector()
    
    # Call another AgenTool
    storage_result = await injector.run('storage', {
        "operation": "read",
        "key": request['storage_key']
    })
    
    # Process with another AgenTool
    processor_result = await injector.run('processor', {
        "operation": "transform",
        "data": storage_result
    })
    
    return {
        "success": True,
        "processed_data": processor_result
    }
```

### 5. Logging and Observability
Integrate with Logfire for monitoring:

```python
import logfire

async def observable_tool(ctx: RunContext[Any], param: str) -> Dict[str, Any]:
    """Tool with integrated observability."""
    with logfire.span('tool_operation', param=param):
        logfire.info(f"Processing parameter: {param}")
        
        result = await process_data(param)
        
        logfire.info(f"Operation completed", result=result)
        return {"success": True, "result": result}
```

## Advanced Features

### 1. Custom Dependencies
Define custom dependency types for your AgenTool:

```python
from dataclasses import dataclass

@dataclass
class StorageDeps:
    database: DatabaseConnection
    cache: CacheClient
    config: StorageConfig

async def advanced_tool(ctx: RunContext[StorageDeps], key: str) -> Dict[str, Any]:
    # Access injected dependencies
    db_result = await ctx.deps.database.query(key)
    cached = await ctx.deps.cache.get(key)
    
    return {"db": db_result, "cached": cached}

# Create with dependencies
agent = create_agentool(
    name='advanced_storage',
    input_schema=StorageInput,
    routing_config=routing,
    tools=[advanced_tool],
    deps_type=StorageDeps
)
```

### 2. Multi-Layer Agent Architecture
Build complex systems with layered AgenTools:

```python
# Layer 1: Low-level storage
storage_agent = create_agentool(
    name='storage',
    input_schema=StorageInput,
    routing_config=storage_routing,
    tools=[read_value, write_value]
)

# Layer 2: Business logic
async def process_business_logic(ctx: RunContext[Any], data: Dict[str, Any]) -> Dict[str, Any]:
    injector = get_injector()
    
    # Read from storage
    stored = await injector.run('storage', {
        "operation": "read",
        "key": data['id']
    })
    
    # Process
    processed = transform_data(stored)
    
    # Write back
    await injector.run('storage', {
        "operation": "write",
        "key": f"processed_{data['id']}",
        "value": processed
    })
    
    return {"success": True, "processed_id": f"processed_{data['id']}"}

business_agent = create_agentool(
    name='business',
    input_schema=BusinessInput,
    routing_config=business_routing,
    tools=[process_business_logic]
)

# Layer 3: API gateway
async def handle_api_request(ctx: RunContext[Any], request: Dict[str, Any]) -> Dict[str, Any]:
    injector = get_injector()
    
    # Validate and route to business layer
    result = await injector.run('business', {
        "operation": request['action'],
        "data": request['payload']
    })
    
    return {"api_response": result, "status": "ok"}
```

### 3. Schema Evolution
Handle schema changes gracefully:

```python
from typing import Union
from pydantic import field_validator

# Version 1
class InputV1(BaseModel):
    operation: Literal['read', 'write']
    key: str
    value: Optional[str] = None

# Version 2 (backward compatible)
class InputV2(BaseModel):
    operation: Literal['read', 'write', 'delete', 'list']
    key: Optional[str] = None  # Made optional for 'list'
    value: Optional[str] = None
    prefix: Optional[str] = None  # New field
    
    @field_validator('key')
    def validate_key(cls, v, info):
        # Ensure key is provided for operations that need it
        if info.data.get('operation') in ['read', 'write', 'delete'] and not v:
            raise ValueError(f"key is required for {info.data['operation']}")
        return v
```

## Testing Guidelines

### 1. Test Structure
AgenTools follow a class-based testing pattern with setup methods:

```python
import asyncio
import json
from agentool.core.injector import get_injector
from agentool.core.registry import AgenToolRegistry

class TestStorageAgent:
    """Test suite for storage AgenTool."""
    
    def setup_method(self):
        """Clear registry and injector before each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        # Import and create the agent
        from my_module import create_storage_agent, _storage
        
        # Clear any global state
        _storage.clear()
        
        # Create the agent
        agent = create_storage_agent()
    
    def test_read_value(self):
        """Test read operation."""
        
        async def run_test():
            injector = get_injector()
            
            # Setup test data
            _storage['test_key'] = 'test_value'
            
            # Test successful read
            result = await injector.run('storage', {
                "operation": "read",
                "key": "test_key"
            })
            
            if hasattr(result, 'output'):
                data = json.loads(result.output)
            else:
                data = result
            
            assert data['success'] is True
            assert data['data'] == 'test_value'
            
            # Test missing key - should raise exception
            try:
                result = await injector.run('storage', {
                    "operation": "read", 
                    "key": "missing_key"
                })
                assert False, "Expected KeyError to be raised for missing key"
            except KeyError as e:
                assert 'not found' in str(e)
        
        asyncio.run(run_test())
```

### 2. Integration Testing
Test the complete AgenTool through the injector:

```python
class TestStorageIntegration:
    """Integration tests for storage AgenTool."""
    
    def setup_method(self):
        """Setup for each test."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        from my_module import create_storage_agent
        agent = create_storage_agent()
    
    def test_storage_workflow(self):
        """Test complete storage workflow."""
        
        async def run_test():
            injector = get_injector()
            
            # Test write operation
            write_result = await injector.run('storage', {
                "operation": "write",
                "key": "test",
                "value": "hello"
            })
            
            if hasattr(write_result, 'output'):
                write_data = json.loads(write_result.output)
            else:
                write_data = write_result
                
            assert write_data['success'] is True
            
            # Test read operation
            read_result = await injector.run('storage', {
                "operation": "read",
                "key": "test"
            })
            
            if hasattr(read_result, 'output'):
                read_data = json.loads(read_result.output)
            else:
                read_data = read_result
                
            assert read_data['data'] == "hello"
            
            # Test list operation
            list_result = await injector.run('storage', {
                "operation": "list",
                "prefix": ""
            })
            
            if hasattr(list_result, 'output'):
                list_data = json.loads(list_result.output)
            else:
                list_data = list_result
                
            assert "test" in list_data['data']['keys']
        
        asyncio.run(run_test())
```

### 3. Schema Validation Testing
Test input validation and error handling:

```python
class TestStorageValidation:
    """Test schema validation for storage AgenTool."""
    
    def setup_method(self):
        """Setup for validation tests."""
        AgenToolRegistry.clear()
        get_injector().clear()
        
        from my_module import create_storage_agent
        agent = create_storage_agent()
    
    def test_input_validation(self):
        """Test input schema validation."""
        
        async def run_test():
            injector = get_injector()
            
            # Test invalid operation
            try:
                result = await injector.run('storage', {
                    "operation": "invalid_op",
                    "key": "test"
                })
                # Should get validation error
                if hasattr(result, 'output'):
                    assert "validation error" in result.output.lower()
            except Exception as e:
                assert "validation error" in str(e).lower()
            
            # Test missing required fields - should raise exception
            try:
                result = await injector.run('storage', {
                    "operation": "write",
                    "key": "test"
                    # missing value - should raise ValueError
                })
                assert False, "Expected ValueError to be raised for missing value"
            except ValueError as e:
                assert 'value' in str(e).lower()
        
        asyncio.run(run_test())
```

### 4. Output Type Testing
Verify output type validation:

```python
def test_output_type_inference():
    """Test that output types are correctly inferred."""
    from agentool.factory import extract_tool_metadata, infer_output_type
    
    metadata = extract_tool_metadata(read_value)
    assert metadata.return_type_annotation == StorageOutput
    
    # Test inference with multiple tools
    tools_metadata = [
        extract_tool_metadata(read_value),
        extract_tool_metadata(write_value)
    ]
    
    inferred = infer_output_type(tools_metadata)
    assert inferred == StorageOutput
```

## Common Patterns

### 1. CRUD Operations
Standard pattern for data management:

```python
from agentool.base import BaseOperationInput

class CRUDInput(BaseOperationInput):
    operation: Literal['create', 'read', 'update', 'delete', 'list']
    id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    filter: Optional[Dict[str, Any]] = None

crud_routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'create': ('create_item', lambda x: {'data': x.data}),
        'read': ('read_item', lambda x: {'id': x.id}),
        'update': ('update_item', lambda x: {'id': x.id, 'data': x.data}),
        'delete': ('delete_item', lambda x: {'id': x.id}),
        'list': ('list_items', lambda x: {'filter': x.filter or {}})
    }
)
```

### 2. Pipeline Processing
Chain operations together:

```python
from agentool.base import BaseOperationInput

class PipelineInput(BaseOperationInput):
    operation: Literal['process']
    pipeline: List[str]  # List of stages
    data: Any

async def process_pipeline(ctx: RunContext[Any], pipeline: List[str], data: Any) -> Dict[str, Any]:
    injector = get_injector()
    current_data = data
    
    for stage in pipeline:
        result = await injector.run(stage, {
            "operation": "transform",
            "data": current_data
        })
        current_data = result.get('output', current_data)
    
    return {"final_result": current_data, "stages_completed": len(pipeline)}
```

### 3. Event-Driven Pattern
React to events:

```python
from agentool.base import BaseOperationInput

class EventInput(BaseOperationInput):
    operation: Literal['emit', 'subscribe', 'unsubscribe']
    event_type: str
    payload: Optional[Any] = None
    handler_id: Optional[str] = None

async def emit_event(ctx: RunContext[Any], event_type: str, payload: Any) -> Dict[str, Any]:
    # Notify all subscribers
    subscribers = get_subscribers(event_type)
    results = []
    
    injector = get_injector()
    for subscriber in subscribers:
        result = await injector.run(subscriber, {
            "operation": "handle",
            "event": event_type,
            "payload": payload
        })
        results.append(result)
    
    return {
        "event": event_type,
        "notified": len(results),
        "results": results
    }
```

### 4. Caching Pattern
Add caching to any AgenTool:

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedToolDeps:
    cache: Dict[str, Tuple[Any, datetime]] = {}
    ttl: timedelta = timedelta(minutes=5)

async def cached_read(ctx: RunContext[CachedToolDeps], key: str) -> Dict[str, Any]:
    # Check cache first
    if key in ctx.deps.cache:
        value, timestamp = ctx.deps.cache[key]
        if datetime.now() - timestamp < ctx.deps.ttl:
            return {"value": value, "cached": True}
    
    # Cache miss - fetch from source
    value = await fetch_from_source(key)
    ctx.deps.cache[key] = (value, datetime.now())
    
    return {"value": value, "cached": False}
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Unknown operation" errors
**Problem**: Operation not found in routing configuration
**Solution**: Ensure operation is defined in Literal type and routing_config

```python
# Bad
class Input(BaseModel):
    operation: str  # Too permissive

# Good  
class Input(BaseModel):
    operation: Literal['read', 'write']  # Explicit options
```

#### 2. Output type validation failures
**Problem**: Tool returns type that doesn't match output_type
**Solution**: Ensure all tools return compatible types

```python
# If output_type is MyOutput, all tools should return:
# 1. MyOutput instances
# 2. Dicts that can be converted to MyOutput
# 3. Or don't specify output_type to use automatic inference

# Note: Automatic inference only works when all tools return 
# the same BaseModel type. For mixed return types, specify
# output_type explicitly or use Dict[str, Any]
```

#### 3. Dependency injection errors
**Problem**: Dependencies not available in context
**Solution**: Ensure deps_type is specified when creating agent

```python
agent = create_agentool(
    name='my_agent',
    input_schema=MyInput,
    routing_config=routing,
    tools=[my_tool],
    deps_type=MyDeps  # Don't forget this!
)
```

#### 4. JSON serialization issues
**Problem**: Complex types not serializable
**Solution**: Use Pydantic models or simple types

```python
# Bad
return {"data": some_complex_object}

# Good
return {"data": some_complex_object.model_dump()}
```

### Debug Tips

1. **Enable Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Check Registry**:
```python
from agentool.core.registry import AgenToolRegistry

# List all registered agents
print(AgenToolRegistry.list_names())

# Check specific agent config
config = AgenToolRegistry.get('my_agent')
print(config.routing_config.operation_map)
```

3. **Test Tool Functions Directly**:
```python
# Test without the agent wrapper
ctx = Mock()
result = await my_tool(ctx, param1="test")
print(result)
```

4. **Validate Schemas**:
```python
# Test input validation
try:
    input_obj = MyInput(operation="invalid")
except ValidationError as e:
    print(e.errors())
```

## Conclusion

AgenTools provide a powerful pattern for building structured, type-safe agents. By following these guidelines, you can create maintainable, testable, and composable agent systems that integrate seamlessly with the pydantic-ai ecosystem.

Remember:
- Start with clear schemas
- Keep tools focused and simple
- Use type hints everywhere
- Test thoroughly
- Document your AgenTools

Happy crafting! 🛠️