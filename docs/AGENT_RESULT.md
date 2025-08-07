# AgenTool Type Safety Documentation

## Overview
This document provides guidelines for AgenTool type safety patterns and best practices. All AgenTools use typed output by default (`use_typed_output=True`), providing full type safety and excellent developer experience.

## AgenTool Status

### Tier 1 - Core Storage (No dependencies)
| AgenTool | Output Type | Features |
|----------|-------------|----------|
| storage_fs | StorageFsOutput | Typed output, discovery patterns, field validators |
| storage_vector | StorageVectorOutput | Typed output, discovery patterns, field validators |
| crypto | CryptoOutput | Typed output, discovery patterns, validators |
| http | HttpOutput | Typed output, validators, authentication support |

### Tier 2 - Storage with KV dependency
| AgenTool | Output Type | Features |
|----------|-------------|----------|
| storage_kv | StorageKvOutput | Typed output, discovery patterns, validators |
| config | ConfigOutput | Typed output, discovery patterns, validators |
| metrics | MetricsOutput | Typed output, discovery patterns |

### Tier 3 - System Tools
| AgenTool | Output Type | Features |
|----------|-------------|----------|
| logging | LoggingOutput | Typed output, structured logging |
| templates | TemplatesOutput | Typed output, Jinja2 rendering |
| session | SessionOutput | Typed output, session management |
| auth | AuthOutput | Typed output, multi-factor auth support |

### Tier 4 - Advanced Tools
| AgenTool | Output Type | Features |
|----------|-------------|----------|
| queue | QueueOutput | Typed output, message queuing |
| management | ManagementOutput | Typed output, agent management |
| scheduler | SchedulerOutput | Typed output, cron/interval scheduling |
| Document/RAG/Graph | - | Deprecated (moved to tmp/) |

### Tier 5 - Workflow Tools
| AgenTool | Output Type | Features |
|----------|-------------|----------|
| workflow_analyzer | WorkflowAnalyzerOutput | Typed output, LLM-powered analysis |
| workflow_specifier | WorkflowSpecifierOutput | Typed output, specification generation |
| workflow_crafter | WorkflowCrafterOutput | Typed output, code generation |
| workflow_evaluator | WorkflowEvaluatorOutput | Typed output, quality evaluation |
| workflow_test_analyzer | WorkflowTestAnalyzerOutput | Typed output, test analysis |
| workflow_test_stubber | WorkflowTestStubberOutput | Typed output, test stub generation |
| workflow_test_crafter | WorkflowTestCrafterOutput | Typed output, test implementation |

## Guidelines for Discovery Operations

### Success/Failure Pattern (HTTP-like semantics)

Discovery operations should follow these patterns:

#### Success=True (HTTP 200-like)
Operation completed successfully with data:
```python
return OutputType(
    success=True,
    operation="verify_hash",
    message="Hash verification successful",
    data={
        "valid": True,  # For verification operations
        "exists": True,  # For existence checks
        "value": data    # For retrieval operations
    }
)
```

#### Success=False (HTTP 404-like)
Resource not found or verification failed (NOT an error):
```python
return OutputType(
    success=False,
    operation="get",
    message="Key not found",
    data=None  # Or minimal data like {"exists": False}
)
```

#### Exceptions (HTTP 500-like)
Actual errors that prevent operation:
```python
raise ValueError("Invalid input format")
raise RuntimeError("Failed to connect to storage backend")
```

### Discovery Operations by AgenTool

| AgenTool | Discovery Operations | Success=False Cases |
|----------|---------------------|-------------------|
| storage_fs | exists | File/directory doesn't exist |
| storage_kv | get, exists, keys | Key not found, no matches |
| storage_vector | search, collection_exists | No vectors found, collection doesn't exist |
| crypto | verify_hash, verify_signature, verify_jwt | Invalid hash/signature/token |
| auth | verify_token, check_permission | Invalid token, no permission |
| session | get_session, validate | Session not found, invalid |
| config | get_config | Config key not found |

## Validation Requirements

### Input Schema Validation

All input schemas should include field validators for operation-specific requirements:

```python
from pydantic import field_validator

class ToolInput(BaseOperationInput):
    operation: Literal['read', 'write', 'delete']
    path: Optional[str] = None
    content: Optional[str] = None
    
    @field_validator('path')
    def validate_path(cls, v, info):
        operation = info.data.get('operation')
        if operation in ['read', 'write', 'delete'] and not v:
            raise ValueError(f"path is required for {operation}")
        return v
    
    @field_validator('content')
    def validate_content(cls, v, info):
        operation = info.data.get('operation')
        if operation == 'write' and v is None:
            raise ValueError("content is required for write operation")
        return v
```

### Validation Matrix

For each AgenTool, define a validation matrix:

| Operation | Required Fields | Optional Fields | Validations |
|-----------|----------------|-----------------|-------------|
| read | path | encoding | path must exist |
| write | path, content | encoding, mode | parent directory must exist |
| delete | path | recursive | - |

## Development Checklist

When creating a new AgenTool:

- [ ] Define Output type (BaseModel) with appropriate fields
- [ ] Define Input schema with field validators
- [ ] Implement discovery operations using success=True/False pattern
- [ ] Write tests using typed result patterns
- [ ] Document the output schema and usage examples
- [ ] Add comprehensive field validation
- [ ] Ensure all operations return typed outputs

## Test Patterns

### Standard Pattern (Typed Output)
```python
result = await injector.run('tool', {...})
assert isinstance(result, ToolOutput)
assert result.success is True
assert result.data['value'] == expected_value
```

### Direct Field Access
```python
# All tools return typed outputs with direct field access
result = await injector.run('storage_kv', {
    'operation': 'get',
    'key': 'user:123'
})
assert result.success is True
assert result.data['exists'] is True
user_data = result.data['value']
```

## Exception Handling Best Practices

### ✅ Correct: Raise for actual errors
```python
if not os.path.exists(parent_dir):
    raise FileNotFoundError(f"Parent directory not found: {parent_dir}")
```

### ✅ Correct: Return success=False for discovery
```python
if key not in storage:
    return StorageOutput(
        success=False,
        operation="get",
        message=f"Key not found: {key}",
        data=None
    )
```

### ❌ Incorrect: Raising for discovery
```python
if key not in storage:
    raise KeyError(f"Key not found: {key}")  # Don't do this for discovery
```

## Workflow Tools Guidelines

### Special Patterns for Workflow AgenTools

Workflow tools have unique characteristics and patterns:

#### 1. Cognitive Data Model Imports
Workflow tools import data models from `agents.models` to provide IDE type understanding:

```python
# Import the data models for cognitive understanding
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from agents.models import AnalyzerOutput, MissingToolSpec
```

#### 2. No Discovery Operations
Unlike storage/system tools, workflows don't have "not found" scenarios:
- They create, transform, or analyze data rather than retrieve it
- All operations should return `success=True` on completion
- Exceptions should only be raised for genuine LLM or infrastructure errors

#### 3. LLM Integration Patterns
Workflow tools use pydantic-ai agents with structured outputs:

```python
# Create LLM agent for analysis
agent = Agent(
    model,
    output_type=AnalyzerOutput,  # Cognitive data model
    system_prompt=system_prompt
)

# Generate analysis using LLM
result = await agent.run(user_prompt)
analysis = result.output  # Direct access to structured output
```

#### 4. State Management
Workflows store intermediate results for continuity:

```python
# Store analysis in storage_kv for later phases
state_key = f'workflow/{workflow_id}/analysis'
await injector.run('storage_kv', {
    'operation': 'set',
    'key': state_key,
    'value': json.dumps(analysis.model_dump())
})
```

### Code Patterns

#### ✅ Correct: Direct Typed Access
```python
# Standard pattern for all AgenTools
result = await injector.run('tool', {...})
assert result.success is True
value = result.data['field_name']
```

#### ✅ Correct: Type Checking
```python
# Verify output type for safety
result = await injector.run('storage_kv', {...})
assert isinstance(result, StorageKvOutput)
if result.success:
    process_data(result.data)
```

#### ✅ Correct: Error Handling
```python
# Handle validation errors appropriately
try:
    result = await injector.run('tool', {...})
    if result.success:
        return result.data['value']
    else:
        # Handle not-found case
        return default_value
except ValueError as e:
    # Handle actual errors
    logger.error(f"Tool error: {e}")
    raise
```

### Workflow Input Schema Pattern

#### Standard Structure
```python
from agentool import BaseOperationInput

class WorkflowToolInput(BaseOperationInput):
    operation: Literal['analyze', 'specify', 'craft', 'evaluate']
    workflow_id: str = Field(description="Unique workflow identifier")
    model: str = Field(default="openai:gpt-4o", description="LLM model to use")
    
    # Operation-specific fields
    task_description: Optional[str] = None
    
    @field_validator('workflow_id')
    def validate_workflow_id(cls, v, info):
        operation = info.data.get('operation')
        if not v:
            raise ValueError("workflow_id is required for all operations")
        return v
```

### Workflow Output Schema Pattern

#### Standard Structure
```python
class WorkflowToolOutput(BaseModel):
    success: bool = Field(description="Whether operation succeeded")
    operation: str = Field(description="Operation that was performed")  # Add for consistency
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(description="Structured results")
    state_ref: str = Field(description="Reference to stored state in storage_kv")
```

### Comprehensive Logging Pattern

#### Add Logging for All Phases
```python
# Log operation start
await injector.run('logging', {
    'operation': 'log',
    'level': 'INFO',
    'logger_name': 'workflow',
    'message': f'{operation.title()} phase started',
    'data': {
        'workflow_id': workflow_id,
        'operation': operation,
        'model': model
    }
})

# Log operation completion
await injector.run('logging', {
    'operation': 'log',
    'level': 'INFO',
    'logger_name': 'workflow',
    'message': f'{operation.title()} phase completed',
    'data': {
        'workflow_id': workflow_id,
        'operation': operation,
        'results_count': len(results)
    }
})
```

### Workflow Development Guidelines

When creating workflow tools:

- [ ] Define output schema inheriting from BaseModel
- [ ] Include `operation` field in output schema
- [ ] Implement comprehensive logging for all phases
- [ ] Import cognitive data models from `agents.models`
- [ ] Write tests expecting typed outputs directly
- [ ] Add logging integration tests
- [ ] Verify LLM integration with structured outputs
- [ ] Store intermediate state in storage_kv for continuity

### Test Pattern for Workflows

#### Standard Test Structure
```python
def test_workflow_operation(self, allow_model_requests):
    """Test workflow operation with LLM integration."""
    
    async def run_test():
        injector = get_injector()
        
        # Run workflow operation
        result = await injector.run('workflow_tool', {
            "operation": "analyze",
            "workflow_id": "test-workflow-001",
            "task_description": "Test task",
            "model": "openai:gpt-4o"
        })
        
        # Verify typed output
        assert result.success is True
        assert result.operation == "analyze"
        assert result.data is not None
        assert result.state_ref.startswith('workflow/')
        
        # Verify state was stored
        state_result = await injector.run('storage_kv', {
            'operation': 'get',
            'key': result.state_ref
        })
        assert state_result.success is True
        assert state_result.data['exists'] is True
    
    asyncio.run(run_test())
```

## Notes

- Discovery operations should be idempotent and predictable
- Success=False is not an error condition, it's a valid response
- Exceptions should only be raised for genuine errors that prevent operation
- All AgenTools should follow consistent patterns for similar operations
- Type safety ensures better IDE support and runtime validation
- Workflow tools focus on data transformation rather than data retrieval
- Cognitive data model imports enhance IDE understanding of data flow