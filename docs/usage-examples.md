# AgenTool Usage Examples and Patterns

This document provides comprehensive examples and patterns for using the AgenTool framework, from simple use cases to complex multi-agent architectures.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Common Patterns](#common-patterns)
- [Advanced Use Cases](#advanced-use-cases)
- [Integration Patterns](#integration-patterns)
- [Testing Patterns](#testing-patterns)
- [Performance Patterns](#performance-patterns)

## Basic Examples

### 1. Simple Calculator

A basic mathematical calculator demonstrating core AgenTool concepts.

```python
from agentool import create_agentool, RoutingConfig, BaseOperationInput
from pydantic import BaseModel, Field
from pydantic_ai import RunContext
from typing import Any, Literal

# Define input schema
class CalculatorInput(BaseOperationInput):
    operation: Literal['add', 'subtract', 'multiply', 'divide']
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

# Define tool functions
async def add(ctx: RunContext[Any], a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

async def subtract(ctx: RunContext[Any], a: float, b: float) -> float:
    """Subtract second number from first."""
    return a - b

async def multiply(ctx: RunContext[Any], a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

async def divide(ctx: RunContext[Any], a: float, b: float) -> float:
    """Divide first number by second."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Configure routing
routing = RoutingConfig(
    operation_field='operation',
    operation_map={
        'add': ('add', lambda x: {'a': x.a, 'b': x.b}),
        'subtract': ('subtract', lambda x: {'a': x.a, 'b': x.b}),
        'multiply': ('multiply', lambda x: {'a': x.a, 'b': x.b}),
        'divide': ('divide', lambda x: {'a': x.a, 'b': x.b})
    }
)

# Create the AgenTool
calculator = create_agentool(
    name='calculator',
    input_schema=CalculatorInput,
    routing_config=routing,
    tools=[add, subtract, multiply, divide],
    description="Mathematical calculator with basic operations",
    tags=['math', 'calculator']
)

# Usage
async def main():
    result = await calculator.run('{"operation": "add", "a": 5, "b": 3}')
    print(f"5 + 3 = {result.output}")  # "8.0"
    
    result = await calculator.run('{"operation": "multiply", "a": 4, "b": 7}')
    print(f"4 * 7 = {result.output}")  # "28.0"
```

### 2. Key-Value Storage

A simple storage system demonstrating CRUD operations.

```python
from agentool import create_agentool, RoutingConfig, BaseOperationInput
from typing import Literal, Optional, Any, Dict
from pydantic_ai import RunContext

# In-memory storage
storage_backend: Dict[str, Any] = {}

class StorageInput(BaseOperationInput):
    operation: Literal['get', 'set', 'delete', 'list']
    key: Optional[str] = None
    value: Optional[Any] = None

async def storage_get(ctx: RunContext[Any], key: str) -> Any:
    """Retrieve a value by key."""
    if key in storage_backend:
        return storage_backend[key]
    return f"Key '{key}' not found"

async def storage_set(ctx: RunContext[Any], key: str, value: Any) -> str:
    """Store a value with a key."""
    storage_backend[key] = value
    return f"Stored value at key '{key}'"

async def storage_delete(ctx: RunContext[Any], key: str) -> str:
    """Delete a key-value pair."""
    if key in storage_backend:
        del storage_backend[key]
        return f"Deleted key '{key}'"
    return f"Key '{key}' not found"

async def storage_list(ctx: RunContext[Any]) -> list[str]:
    """List all keys."""
    return list(storage_backend.keys())

routing = RoutingConfig(
    operation_map={
        'get': ('storage_get', lambda x: {'key': x.key}),
        'set': ('storage_set', lambda x: {'key': x.key, 'value': x.value}),
        'delete': ('storage_delete', lambda x: {'key': x.key}),
        'list': ('storage_list', lambda x: {}),
    }
)

storage = create_agentool(
    name='kv_storage',
    input_schema=StorageInput,
    routing_config=routing,
    tools=[storage_get, storage_set, storage_delete, storage_list],
    description="Key-value storage system"
)

# Usage
async def storage_demo():
    # Store some values
    await storage.run('{"operation": "set", "key": "user:123", "value": {"name": "Alice", "age": 30}}')
    await storage.run('{"operation": "set", "key": "config:timeout", "value": 300}')
    
    # Retrieve values
    result = await storage.run('{"operation": "get", "key": "user:123"}')
    print(f"User data: {result.output}")
    
    # List all keys
    result = await storage.run('{"operation": "list"}')
    print(f"All keys: {result.output}")
```

### 3. Text Processing

Text manipulation operations with various transformations.

```python
import re
from typing import Literal, Optional

class TextInput(BaseOperationInput):
    operation: Literal['uppercase', 'lowercase', 'reverse', 'count_words', 'replace']
    text: str
    pattern: Optional[str] = None
    replacement: Optional[str] = None

async def text_uppercase(ctx: RunContext[Any], text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()

async def text_lowercase(ctx: RunContext[Any], text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()

async def text_reverse(ctx: RunContext[Any], text: str) -> str:
    """Reverse the text."""
    return text[::-1]

async def text_count_words(ctx: RunContext[Any], text: str) -> int:
    """Count words in text."""
    return len(text.split())

async def text_replace(ctx: RunContext[Any], text: str, pattern: str, replacement: str) -> str:
    """Replace pattern in text with replacement."""
    return re.sub(pattern, replacement, text)

routing = RoutingConfig(
    operation_map={
        'uppercase': ('text_uppercase', lambda x: {'text': x.text}),
        'lowercase': ('text_lowercase', lambda x: {'text': x.text}),
        'reverse': ('text_reverse', lambda x: {'text': x.text}),
        'count_words': ('text_count_words', lambda x: {'text': x.text}),
        'replace': ('text_replace', lambda x: {
            'text': x.text, 
            'pattern': x.pattern, 
            'replacement': x.replacement
        }),
    }
)

text_processor = create_agentool(
    name='text_processor',
    input_schema=TextInput,
    routing_config=routing,
    tools=[text_uppercase, text_lowercase, text_reverse, text_count_words, text_replace],
    description="Text processing utilities"
)
```

## Common Patterns

### 1. Structured Output with Validation

Using Pydantic models for structured, validated outputs.

```python
from datetime import datetime
from pydantic import BaseModel

class ProcessingResult(BaseModel):
    """Structured result with validation."""
    success: bool
    result: Any
    timestamp: datetime
    metadata: Dict[str, Any]

class DataInput(BaseOperationInput):
    operation: Literal['process', 'validate', 'transform']
    data: Dict[str, Any]
    options: Optional[Dict[str, str]] = None

async def process_data(ctx: RunContext[Any], data: Dict[str, Any], options: Dict[str, str]) -> ProcessingResult:
    """Process data and return structured result."""
    try:
        # Simulate processing
        processed = {k: str(v).upper() for k, v in data.items()}
        
        return ProcessingResult(
            success=True,
            result=processed,
            timestamp=datetime.now(),
            metadata={"processed_fields": len(data), "options": options}
        )
    except Exception as e:
        return ProcessingResult(
            success=False,
            result=str(e),
            timestamp=datetime.now(),
            metadata={"error": True}
        )

async def validate_data(ctx: RunContext[Any], data: Dict[str, Any]) -> ProcessingResult:
    """Validate data structure."""
    required_fields = ['id', 'name']
    missing = [field for field in required_fields if field not in data]
    
    return ProcessingResult(
        success=len(missing) == 0,
        result={"missing_fields": missing} if missing else "Valid",
        timestamp=datetime.now(),
        metadata={"validation": "complete"}
    )

# Create with structured output type
data_processor = create_agentool(
    name='data_processor',
    input_schema=DataInput,
    routing_config=RoutingConfig(
        operation_map={
            'process': ('process_data', lambda x: {'data': x.data, 'options': x.options or {}}),
            'validate': ('validate_data', lambda x: {'data': x.data}),
        }
    ),
    tools=[process_data, validate_data],
    output_type=ProcessingResult,  # Automatic validation and serialization
    description="Data processing with structured output"
)
```

### 2. Error Handling Patterns

Comprehensive error handling and recovery strategies.

```python
class SafeOperationInput(BaseOperationInput):
    operation: Literal['safe_divide', 'safe_parse', 'safe_request']
    a: Optional[float] = None
    b: Optional[float] = None
    text: Optional[str] = None
    url: Optional[str] = None

class SafeResult(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

async def safe_divide(ctx: RunContext[Any], a: float, b: float) -> SafeResult:
    """Division with error handling."""
    try:
        if b == 0:
            return SafeResult(
                success=False,
                error="Division by zero",
                error_type="ZeroDivisionError"
            )
        return SafeResult(success=True, result=a / b)
    except Exception as e:
        return SafeResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__
        )

async def safe_parse(ctx: RunContext[Any], text: str) -> SafeResult:
    """JSON parsing with error handling."""
    try:
        import json
        result = json.loads(text)
        return SafeResult(success=True, result=result)
    except json.JSONDecodeError as e:
        return SafeResult(
            success=False,
            error=f"Invalid JSON: {e}",
            error_type="JSONDecodeError"
        )
    except Exception as e:
        return SafeResult(
            success=False,
            error=str(e),
            error_type=type(e).__name__
        )

safe_operations = create_agentool(
    name='safe_operations',
    input_schema=SafeOperationInput,
    routing_config=RoutingConfig(
        operation_map={
            'safe_divide': ('safe_divide', lambda x: {'a': x.a, 'b': x.b}),
            'safe_parse': ('safe_parse', lambda x: {'text': x.text}),
        }
    ),
    tools=[safe_divide, safe_parse],
    output_type=SafeResult,
    description="Safe operations with comprehensive error handling"
)
```

### 3. Dynamic Routing

Advanced routing based on input content or context.

```python
class DynamicInput(BaseOperationInput):
    operation: Literal['route_by_type', 'route_by_size', 'route_by_content']
    data: Any
    threshold: Optional[int] = 100

async def handle_string(ctx: RunContext[Any], data: str) -> str:
    """Handle string data."""
    return f"Processed string: {data.upper()}"

async def handle_number(ctx: RunContext[Any], data: Union[int, float]) -> str:
    """Handle numeric data."""
    return f"Processed number: {data * 2}"

async def handle_list(ctx: RunContext[Any], data: list) -> str:
    """Handle list data."""
    return f"Processed list with {len(data)} items"

async def handle_large_data(ctx: RunContext[Any], data: Any) -> str:
    """Handle large datasets."""
    return f"Processed large data (size: {len(str(data))})"

async def handle_small_data(ctx: RunContext[Any], data: Any) -> str:
    """Handle small datasets."""
    return f"Processed small data: {data}"

def dynamic_transform(x: DynamicInput) -> Dict[str, Any]:
    """Dynamic transformation based on data type and size."""
    data = x.data
    
    # Route by type
    if x.operation == 'route_by_type':
        if isinstance(data, str):
            return {'data': data}
        elif isinstance(data, (int, float)):
            return {'data': data}
        elif isinstance(data, list):
            return {'data': data}
    
    # Route by size
    elif x.operation == 'route_by_size':
        data_size = len(str(data))
        if data_size > x.threshold:
            return {'data': data}
        else:
            return {'data': data}
    
    return {'data': data}

# Create multiple routing configurations for different operations
def create_dynamic_routing() -> RoutingConfig:
    """Create routing config with dynamic logic."""
    
    def type_based_router(x: DynamicInput) -> Tuple[str, Dict[str, Any]]:
        """Route based on data type."""
        data = x.data
        if isinstance(data, str):
            return 'handle_string', {'data': data}
        elif isinstance(data, (int, float)):
            return 'handle_number', {'data': data}
        elif isinstance(data, list):
            return 'handle_list', {'data': data}
        else:
            return 'handle_string', {'data': str(data)}
    
    def size_based_router(x: DynamicInput) -> Tuple[str, Dict[str, Any]]:
        """Route based on data size."""
        data_size = len(str(x.data))
        if data_size > x.threshold:
            return 'handle_large_data', {'data': x.data}
        else:
            return 'handle_small_data', {'data': x.data}
    
    # Custom routing logic
    def smart_transform(x: DynamicInput) -> Dict[str, Any]:
        if x.operation == 'route_by_type':
            tool_name, args = type_based_router(x)
        elif x.operation == 'route_by_size':
            tool_name, args = size_based_router(x)
        else:
            tool_name, args = 'handle_string', {'data': str(x.data)}
        
        # Store the tool name in a way the manager can use
        # This is a simplified example - real implementation would need
        # to work with the existing routing system
        return args
    
    return RoutingConfig(
        operation_map={
            'route_by_type': ('handle_string', smart_transform),  # Simplified
            'route_by_size': ('handle_large_data', smart_transform),
            'route_by_content': ('handle_string', smart_transform),
        }
    )

dynamic_router = create_agentool(
    name='dynamic_router',
    input_schema=DynamicInput,
    routing_config=create_dynamic_routing(),
    tools=[handle_string, handle_number, handle_list, handle_large_data, handle_small_data],
    description="Dynamic routing based on input characteristics"
)
```

## Advanced Use Cases

### 1. Multi-Agent Workflow

Complex workflow with multiple interacting agents using dependency injection.

```python
from agentool.core.injector import get_injector, InjectedDeps
from dataclasses import dataclass

# Data validation agent
class ValidationInput(BaseOperationInput):
    operation: Literal['validate_user', 'validate_order']
    data: Dict[str, Any]

async def validate_user(ctx: RunContext[Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate user data."""
    required = ['id', 'name', 'email']
    missing = [field for field in required if field not in data]
    
    return {
        'valid': len(missing) == 0,
        'missing_fields': missing,
        'data': data
    }

async def validate_order(ctx: RunContext[Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate order data."""
    required = ['user_id', 'items', 'total']
    missing = [field for field in required if field not in data]
    
    return {
        'valid': len(missing) == 0,
        'missing_fields': missing,
        'data': data
    }

validator = create_agentool(
    name='validator',
    input_schema=ValidationInput,
    routing_config=RoutingConfig(
        operation_map={
            'validate_user': ('validate_user', lambda x: {'data': x.data}),
            'validate_order': ('validate_order', lambda x: {'data': x.data}),
        }
    ),
    tools=[validate_user, validate_order],
    description="Data validation service"
)

# Processing agent that depends on validator
class ProcessingInput(BaseOperationInput):
    operation: Literal['process_user', 'process_order']
    data: Dict[str, Any]

@dataclass
class ProcessingDeps(InjectedDeps):
    """Dependencies for the processing agent."""
    
    async def validate_and_process_user(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and process user data."""
        # First validate
        validation_result = await self.call_agent('validator', {
            'operation': 'validate_user',
            'data': data
        })
        
        validation_data = json.loads(validation_result.output)
        if not validation_data['valid']:
            return {
                'success': False,
                'error': f"Validation failed: {validation_data['missing_fields']}"
            }
        
        # Then process
        processed_data = {
            **data,
            'processed_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        return {
            'success': True,
            'result': processed_data
        }

async def process_user(ctx: RunContext[ProcessingDeps], data: Dict[str, Any]) -> Dict[str, Any]:
    """Process user data with validation."""
    return await ctx.deps.validate_and_process_user(data)

async def process_order(ctx: RunContext[ProcessingDeps], data: Dict[str, Any]) -> Dict[str, Any]:
    """Process order data with validation."""
    # Similar pattern for orders
    validation_result = await ctx.deps.call_agent('validator', {
        'operation': 'validate_order',
        'data': data
    })
    
    validation_data = json.loads(validation_result.output)
    if not validation_data['valid']:
        return {
            'success': False,
            'error': f"Order validation failed: {validation_data['missing_fields']}"
        }
    
    # Process the order
    processed_order = {
        **data,
        'processed_at': datetime.now().isoformat(),
        'status': 'pending'
    }
    
    return {
        'success': True,
        'result': processed_order
    }

processor = create_agentool(
    name='processor',
    input_schema=ProcessingInput,
    routing_config=RoutingConfig(
        operation_map={
            'process_user': ('process_user', lambda x: {'data': x.data}),
            'process_order': ('process_order', lambda x: {'data': x.data}),
        }
    ),
    tools=[process_user, process_order],
    deps_type=ProcessingDeps,
    dependencies=['validator'],  # Declares dependency
    description="Data processing service with validation"
)

# Usage
async def workflow_demo():
    injector = get_injector()
    
    # Process user data
    user_data = {
        'id': 'user123',
        'name': 'Alice',
        'email': 'alice@example.com'
    }
    
    result = await injector.run('processor', {
        'operation': 'process_user',
        'data': user_data
    })
    
    print(f"User processing result: {result.output}")
```

### 2. Pipeline Processing

Sequential processing through multiple agents.

```python
class PipelineInput(BaseOperationInput):
    operation: Literal['extract', 'transform', 'load', 'full_pipeline']
    data: Any
    config: Optional[Dict[str, Any]] = None

# Extract agent
async def extract_data(ctx: RunContext[Any], data: Any) -> Dict[str, Any]:
    """Extract data from various sources."""
    if isinstance(data, str) and data.startswith('http'):
        # Simulate fetching from URL
        return {'source': 'url', 'content': f"Content from {data}", 'extracted_at': datetime.now().isoformat()}
    elif isinstance(data, dict):
        return {'source': 'dict', 'content': data, 'extracted_at': datetime.now().isoformat()}
    else:
        return {'source': 'raw', 'content': str(data), 'extracted_at': datetime.now().isoformat()}

extractor = create_agentool(
    name='extractor',
    input_schema=PipelineInput,
    routing_config=RoutingConfig(
        operation_map={
            'extract': ('extract_data', lambda x: {'data': x.data}),
        }
    ),
    tools=[extract_data],
    description="Data extraction service"
)

# Transform agent
async def transform_data(ctx: RunContext[Any], data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform extracted data."""
    content = data.get('content', {})
    
    # Apply transformations based on config
    if config.get('uppercase', False):
        if isinstance(content, str):
            content = content.upper()
        elif isinstance(content, dict):
            content = {k: str(v).upper() if isinstance(v, str) else v for k, v in content.items()}
    
    return {
        'original': data,
        'transformed_content': content,
        'transformed_at': datetime.now().isoformat(),
        'transformations_applied': list(config.keys())
    }

transformer = create_agentool(
    name='transformer',
    input_schema=PipelineInput,
    routing_config=RoutingConfig(
        operation_map={
            'transform': ('transform_data', lambda x: {'data': x.data, 'config': x.config or {}}),
        }
    ),
    tools=[transform_data],
    description="Data transformation service"
)

# Load agent
async def load_data(ctx: RunContext[Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Load transformed data to destination."""
    # Simulate loading to database/storage
    return {
        'loaded': True,
        'data': data,
        'loaded_at': datetime.now().isoformat(),
        'destination': 'database'
    }

loader = create_agentool(
    name='loader',
    input_schema=PipelineInput,
    routing_config=RoutingConfig(
        operation_map={
            'load': ('load_data', lambda x: {'data': x.data}),
        }
    ),
    tools=[load_data],
    description="Data loading service"
)

# Pipeline orchestrator
@dataclass
class PipelineDeps(InjectedDeps):
    """Dependencies for pipeline orchestration."""
    
    async def run_full_pipeline(self, data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete ETL pipeline."""
        # Extract
        extract_result = await self.call_agent('extractor', {
            'operation': 'extract',
            'data': data
        })
        extracted = json.loads(extract_result.output)
        
        # Transform
        transform_result = await self.call_agent('transformer', {
            'operation': 'transform',
            'data': extracted,
            'config': config
        })
        transformed = json.loads(transform_result.output)
        
        # Load
        load_result = await self.call_agent('loader', {
            'operation': 'load',
            'data': transformed
        })
        loaded = json.loads(load_result.output)
        
        return {
            'pipeline_complete': True,
            'steps': {
                'extract': extracted,
                'transform': transformed,
                'load': loaded
            },
            'completed_at': datetime.now().isoformat()
        }

async def full_pipeline(ctx: RunContext[PipelineDeps], data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute full pipeline."""
    return await ctx.deps.run_full_pipeline(data, config)

pipeline = create_agentool(
    name='pipeline',
    input_schema=PipelineInput,
    routing_config=RoutingConfig(
        operation_map={
            'full_pipeline': ('full_pipeline', lambda x: {'data': x.data, 'config': x.config or {}}),
        }
    ),
    tools=[full_pipeline],
    deps_type=PipelineDeps,
    dependencies=['extractor', 'transformer', 'loader'],
    description="ETL pipeline orchestrator"
)
```

## Integration Patterns

### 1. Hybrid LLM + AgenTool Systems

Combining LLM intelligence with deterministic operations.

```python
from pydantic_ai import Agent

# Create LLM agent for natural language processing
llm_agent = Agent(
    'openai:gpt-4',
    system_prompt="""You are a helpful assistant that processes natural language 
    queries and converts them to structured operations. When users ask for specific 
    operations like calculations, data storage, or processing, respond with JSON 
    that can be used with our backend systems."""
)

# Create AgenTool for structured operations
structured_ops = create_agentool(
    name='structured_ops',
    input_schema=CalculatorInput,  # From previous example
    routing_config=routing,
    tools=[add, subtract, multiply, divide],
    description="Structured mathematical operations"
)

async def hybrid_processing(user_query: str) -> str:
    """Process user queries using both LLM and AgenTool."""
    
    # First, use LLM to understand intent and extract structure
    llm_prompt = f"""
    User query: "{user_query}"
    
    If this is a mathematical operation, respond with JSON in this format:
    {{"operation": "add|subtract|multiply|divide", "a": number, "b": number}}
    
    If this is not a mathematical operation, respond with a natural language answer.
    """
    
    llm_result = await llm_agent.run(llm_prompt)
    
    # Try to parse as JSON for structured operations
    try:
        import json
        structured_input = json.loads(llm_result.output)
        
        # If it's structured data, use AgenTool
        if 'operation' in structured_input:
            agentool_result = await structured_ops.run(json.dumps(structured_input))
            return f"Calculation result: {agentool_result.output}"
    
    except json.JSONDecodeError:
        # Not structured data, return LLM response
        pass
    
    return llm_result.output

# Usage
async def hybrid_demo():
    # Mathematical queries get processed by AgenTool
    result = await hybrid_processing("What is 15 plus 27?")
    print(result)  # "Calculation result: 42.0"
    
    # General queries get processed by LLM
    result = await hybrid_processing("What is the capital of France?")
    print(result)  # Natural language response from LLM
```

### 2. API Gateway Pattern

Using AgenTool as an API gateway for microservices.

```python
class APIRequest(BaseOperationInput):
    operation: Literal['users', 'orders', 'products', 'auth']
    method: Literal['GET', 'POST', 'PUT', 'DELETE']
    path: str
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None

# Microservice handlers
async def handle_users(ctx: RunContext[Any], method: str, path: str, data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Handle user service requests."""
    # Simulate user service logic
    if method == 'GET':
        return {'users': [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]}
    elif method == 'POST':
        return {'created_user': data, 'id': 123}
    elif method == 'PUT':
        return {'updated_user': data}
    elif method == 'DELETE':
        return {'deleted': True}

async def handle_orders(ctx: RunContext[Any], method: str, path: str, data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Handle order service requests."""
    if method == 'GET':
        return {'orders': [{'id': 1, 'total': 100}, {'id': 2, 'total': 200}]}
    elif method == 'POST':
        return {'created_order': data, 'id': 456}

async def handle_auth(ctx: RunContext[Any], method: str, path: str, data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Handle authentication requests."""
    if method == 'POST' and data.get('username') and data.get('password'):
        return {'token': 'jwt_token_here', 'expires': '2024-12-31T23:59:59'}
    return {'error': 'Invalid credentials'}

# API Gateway AgenTool
api_gateway = create_agentool(
    name='api_gateway',
    input_schema=APIRequest,
    routing_config=RoutingConfig(
        operation_map={
            'users': ('handle_users', lambda x: {
                'method': x.method, 'path': x.path, 
                'data': x.data or {}, 'headers': x.headers or {}
            }),
            'orders': ('handle_orders', lambda x: {
                'method': x.method, 'path': x.path,
                'data': x.data or {}, 'headers': x.headers or {}
            }),
            'auth': ('handle_auth', lambda x: {
                'method': x.method, 'path': x.path,
                'data': x.data or {}, 'headers': x.headers or {}
            }),
        }
    ),
    tools=[handle_users, handle_orders, handle_auth],
    description="API Gateway for microservices"
)

# Usage
async def api_demo():
    # GET users
    result = await api_gateway.run(json.dumps({
        'operation': 'users',
        'method': 'GET',
        'path': '/users',
    }))
    print(f"Users: {result.output}")
    
    # Create order
    result = await api_gateway.run(json.dumps({
        'operation': 'orders',
        'method': 'POST',
        'path': '/orders',
        'data': {'user_id': 1, 'items': ['item1', 'item2'], 'total': 150}
    }))
    print(f"Created order: {result.output}")
```

## Testing Patterns

### 1. Unit Testing AgenTools

Comprehensive testing strategies for individual AgenTools.

```python
import pytest
from unittest.mock import AsyncMock
from agentool.core.injector import get_injector

class TestCalculatorAgenTool:
    """Test suite for calculator AgenTool."""
    
    @pytest.mark.asyncio
    async def test_add_operation(self):
        """Test addition operation."""
        result = await calculator.run('{"operation": "add", "a": 5, "b": 3}')
        assert float(result.output) == 8.0
    
    @pytest.mark.asyncio
    async def test_subtract_operation(self):
        """Test subtraction operation."""
        result = await calculator.run('{"operation": "subtract", "a": 10, "b": 4}')
        assert float(result.output) == 6.0
    
    @pytest.mark.asyncio
    async def test_divide_by_zero(self):
        """Test division by zero error handling."""
        result = await calculator.run('{"operation": "divide", "a": 10, "b": 0}')
        assert "Cannot divide by zero" in result.output
    
    @pytest.mark.asyncio
    async def test_invalid_operation(self):
        """Test invalid operation handling."""
        result = await calculator.run('{"operation": "invalid", "a": 1, "b": 2}')
        assert "Unknown operation" in result.output
    
    @pytest.mark.asyncio
    async def test_invalid_json(self):
        """Test invalid JSON input."""
        result = await calculator.run('invalid json')
        assert "Invalid JSON" in result.output
    
    @pytest.mark.asyncio
    async def test_missing_fields(self):
        """Test missing required fields."""
        result = await calculator.run('{"operation": "add", "a": 5}')
        assert "Error creating input model" in result.output
    
    @pytest.mark.asyncio
    async def test_with_pydantic_input(self):
        """Test with Pydantic model input via injector."""
        injector = get_injector()
        input_model = CalculatorInput(operation="multiply", a=6, b=7)
        
        result = await injector.run('calculator', input_model)
        assert float(result.output) == 42.0

class TestStorageAgenTool:
    """Test suite for storage AgenTool."""
    
    @pytest.fixture(autouse=True)
    async def setup_storage(self):
        """Reset storage before each test."""
        storage_backend.clear()
        yield
        storage_backend.clear()
    
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """Test setting and getting values."""
        # Set a value
        result = await storage.run('{"operation": "set", "key": "test", "value": "hello"}')
        assert "Stored value" in result.output
        
        # Get the value
        result = await storage.run('{"operation": "get", "key": "test"}')
        assert result.output == '"hello"'
    
    @pytest.mark.asyncio
    async def test_list_keys(self):
        """Test listing keys."""
        # Add some keys
        await storage.run('{"operation": "set", "key": "key1", "value": "value1"}')
        await storage.run('{"operation": "set", "key": "key2", "value": "value2"}')
        
        # List keys
        result = await storage.run('{"operation": "list"}')
        keys = json.loads(result.output)
        assert sorted(keys) == ["key1", "key2"]
    
    @pytest.mark.asyncio
    async def test_delete_key(self):
        """Test deleting keys."""
        # Set and delete
        await storage.run('{"operation": "set", "key": "temp", "value": "temp_value"}')
        result = await storage.run('{"operation": "delete", "key": "temp"}')
        assert "Deleted key" in result.output
        
        # Verify it's gone
        result = await storage.run('{"operation": "get", "key": "temp"}')
        assert "not found" in result.output
```

### 2. Integration Testing with Mocks

Testing multi-agent systems with dependency injection.

```python
class TestMultiAgentWorkflow:
    """Test multi-agent workflows with mocking."""
    
    @pytest.fixture
    async def mock_validator(self):
        """Create mock validator agent."""
        from agentool import create_agentool
        
        async def mock_validate_user(ctx, data):
            # Always return valid for testing
            return {'valid': True, 'missing_fields': [], 'data': data}
        
        async def mock_validate_order(ctx, data):
            # Return validation result based on test data
            if 'test_invalid' in data:
                return {'valid': False, 'missing_fields': ['user_id'], 'data': data}
            return {'valid': True, 'missing_fields': [], 'data': data}
        
        return create_agentool(
            name='mock_validator',
            input_schema=ValidationInput,
            routing_config=RoutingConfig(
                operation_map={
                    'validate_user': ('mock_validate_user', lambda x: {'data': x.data}),
                    'validate_order': ('mock_validate_order', lambda x: {'data': x.data}),
                }
            ),
            tools=[mock_validate_user, mock_validate_order],
            description="Mock validator for testing"
        )
    
    @pytest.mark.asyncio
    async def test_successful_user_processing(self, mock_validator):
        """Test successful user processing with mock validator."""
        injector = get_injector()
        
        # Override validator with mock
        with injector.override(validator=mock_validator):
            result = await injector.run('processor', {
                'operation': 'process_user',
                'data': {'id': 'test123', 'name': 'Test User', 'email': 'test@example.com'}
            })
            
            result_data = json.loads(result.output)
            assert result_data['success'] is True
            assert 'result' in result_data
            assert result_data['result']['status'] == 'active'
    
    @pytest.mark.asyncio
    async def test_failed_order_processing(self, mock_validator):
        """Test failed order processing with validation errors."""
        injector = get_injector()
        
        with injector.override(validator=mock_validator):
            result = await injector.run('processor', {
                'operation': 'process_order',
                'data': {'test_invalid': True, 'items': ['item1']}  # Missing user_id
            })
            
            result_data = json.loads(result.output)
            assert result_data['success'] is False
            assert 'validation failed' in result_data['error']
```

### 3. Performance Testing

Testing performance characteristics and optimization.

```python
import time
import asyncio
from statistics import mean, median

class TestPerformance:
    """Performance testing for AgenTools."""
    
    @pytest.mark.asyncio
    async def test_calculator_performance(self):
        """Test calculator performance under load."""
        iterations = 100
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = await calculator.run(f'{{"operation": "add", "a": {i}, "b": {i+1}}}')
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert float(result.output) == i + (i + 1)
        
        avg_time = mean(times)
        median_time = median(times)
        
        print(f"Average response time: {avg_time:.4f}s")
        print(f"Median response time: {median_time:.4f}s")
        
        # Assert reasonable performance (adjust thresholds as needed)
        assert avg_time < 0.01, f"Average response time too high: {avg_time:.4f}s"
        assert median_time < 0.01, f"Median response time too high: {median_time:.4f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations."""
        async def single_operation(i: int):
            return await calculator.run(f'{{"operation": "multiply", "a": {i}, "b": 2}}')
        
        # Run 50 concurrent operations
        start_time = time.time()
        tasks = [single_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all results
        for i, result in enumerate(results):
            assert float(result.output) == i * 2
        
        total_time = end_time - start_time
        print(f"50 concurrent operations completed in {total_time:.4f}s")
        
        # Should handle concurrent operations efficiently
        assert total_time < 1.0, f"Concurrent operations too slow: {total_time:.4f}s"
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage patterns."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(1000):
            await calculator.run(f'{{"operation": "add", "a": {i}, "b": {i}}}')
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase after 1000 operations: {memory_increase / 1024 / 1024:.2f} MB")
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increase too high: {memory_increase / 1024 / 1024:.2f} MB"
```

## Performance Patterns

### 1. Caching and Memoization

Implementing caching for expensive operations.

```python
from functools import lru_cache
import asyncio
from typing import Any, Dict

# Simple async cache decorator
def async_lru_cache(maxsize: int = 128):
    """Async LRU cache decorator."""
    def decorator(func):
        cache = {}
        
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                return cache[key]
            
            result = await func(*args, **kwargs)
            cache[key] = result
            
            # Simple cache size management
            if len(cache) > maxsize:
                # Remove oldest entry (simplified LRU)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            return result
        
        return wrapper
    return decorator

class CachedComputeInput(BaseOperationInput):
    operation: Literal['expensive_calc', 'fibonacci', 'prime_check']
    n: int
    cache_enabled: bool = True

@async_lru_cache(maxsize=100)
async def expensive_calculation(ctx: RunContext[Any], n: int) -> int:
    """Simulate expensive calculation with caching."""
    # Simulate expensive work
    await asyncio.sleep(0.1)  # Simulate 100ms of work
    return n * n * n + n * n + n

@async_lru_cache(maxsize=50)
async def fibonacci(ctx: RunContext[Any], n: int) -> int:
    """Calculate Fibonacci with caching."""
    if n <= 1:
        return n
    
    # This would normally be recursive, but we'll simulate
    await asyncio.sleep(0.01 * n)  # Simulate work proportional to n
    
    # Simplified Fibonacci calculation
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

cached_compute = create_agentool(
    name='cached_compute',
    input_schema=CachedComputeInput,
    routing_config=RoutingConfig(
        operation_map={
            'expensive_calc': ('expensive_calculation', lambda x: {'n': x.n}),
            'fibonacci': ('fibonacci', lambda x: {'n': x.n}),
        }
    ),
    tools=[expensive_calculation, fibonacci],
    description="Compute operations with caching"
)

# Performance comparison
async def cache_demo():
    """Demonstrate caching performance benefits."""
    import time
    
    # First call (cache miss)
    start = time.time()
    result1 = await cached_compute.run('{"operation": "expensive_calc", "n": 100}')
    first_call_time = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    result2 = await cached_compute.run('{"operation": "expensive_calc", "n": 100}')
    second_call_time = time.time() - start
    
    print(f"First call (cache miss): {first_call_time:.4f}s")
    print(f"Second call (cache hit): {second_call_time:.4f}s")
    print(f"Speed improvement: {first_call_time / second_call_time:.2f}x")
    
    assert result1.output == result2.output
    assert second_call_time < first_call_time / 10  # Should be much faster
```

### 2. Batch Processing

Optimizing for batch operations.

```python
class BatchInput(BaseOperationInput):
    operation: Literal['batch_process', 'batch_transform']
    items: list[Dict[str, Any]]
    batch_size: int = 10

async def batch_process(ctx: RunContext[Any], items: list[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
    """Process items in batches for better performance."""
    processed_items = []
    
    # Process in batches
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch concurrently
        async def process_item(item):
            # Simulate processing
            await asyncio.sleep(0.01)  # Simulate work
            return {**item, 'processed': True, 'timestamp': time.time()}
        
        batch_results = await asyncio.gather(*[process_item(item) for item in batch])
        processed_items.extend(batch_results)
    
    return {
        'total_items': len(items),
        'processed_items': len(processed_items),
        'batch_size': batch_size,
        'items': processed_items
    }

async def batch_transform(ctx: RunContext[Any], items: list[Dict[str, Any]], batch_size: int) -> Dict[str, Any]:
    """Transform items in optimized batches."""
    transformed_items = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Batch transformation (more efficient than individual transforms)
        batch_transformed = []
        for item in batch:
            # Apply transformations
            transformed = {
                'id': item.get('id', i),
                'data': str(item.get('data', '')).upper(),
                'processed_at': time.time(),
                'batch_index': i // batch_size
            }
            batch_transformed.append(transformed)
        
        transformed_items.extend(batch_transformed)
    
    return {
        'total_items': len(items),
        'transformed_items': len(transformed_items),
        'batch_size': batch_size,
        'items': transformed_items
    }

batch_processor = create_agentool(
    name='batch_processor',
    input_schema=BatchInput,
    routing_config=RoutingConfig(
        operation_map={
            'batch_process': ('batch_process', lambda x: {'items': x.items, 'batch_size': x.batch_size}),
            'batch_transform': ('batch_transform', lambda x: {'items': x.items, 'batch_size': x.batch_size}),
        }
    ),
    tools=[batch_process, batch_transform],
    description="Batch processing for performance optimization"
)

# Usage
async def batch_demo():
    """Demonstrate batch processing performance."""
    # Create test data
    test_items = [{'id': i, 'data': f'item_{i}'} for i in range(100)]
    
    # Test different batch sizes
    for batch_size in [1, 10, 50]:
        start = time.time()
        result = await batch_processor.run(json.dumps({
            'operation': 'batch_process',
            'items': test_items,
            'batch_size': batch_size
        }))
        elapsed = time.time() - start
        
        result_data = json.loads(result.output)
        print(f"Batch size {batch_size}: {elapsed:.4f}s for {result_data['total_items']} items")
```

This comprehensive set of examples and patterns provides developers with practical guidance for implementing AgenTool-based systems, from simple use cases to complex multi-agent architectures with performance optimization and robust testing strategies.