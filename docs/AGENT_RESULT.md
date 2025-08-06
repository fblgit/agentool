# AgenTool Type Safety Migration Tracking

## Overview
This document tracks the migration of AgenTools to strict typing with the `use_typed_output` flag and establishes guidelines for consistent behavior patterns across all AgenTools.

## Migration Status

### Tier 1 - Core Storage (No dependencies)
| AgenTool | Status | Output Type | Notes |
|----------|--------|-------------|-------|
| storage_fs | ✅ Complete | StorageFsOutput | Migrated with typed output enabled |
| crypto | ✅ Complete | CryptoOutput | Typed output enabled, discovery patterns updated, validators added |
| http | ⏳ Pending | - | Next in queue |

### Tier 2 - Storage with KV dependency
| AgenTool | Status | Output Type | Notes |
|----------|--------|-------------|-------|
| storage_kv | ⏳ Pending | StorageKvOutput | Special handling needed for discovery operations |
| config | ⏳ Pending | - | Depends on storage_kv |
| metrics | ⏳ Pending | - | Depends on storage_kv |

### Tier 3 - System Tools
| AgenTool | Status | Output Type | Notes |
|----------|--------|-------------|-------|
| logging | ⏳ Pending | - | Depends on storage_fs |
| templates | ⏳ Pending | - | Depends on storage_kv, storage_fs |
| session | ⏳ Pending | - | Depends on storage_kv |
| auth | ⏳ Pending | - | Depends on session, storage_kv, crypto |

### Tier 4 - Advanced Tools
| AgenTool | Status | Output Type | Notes |
|----------|--------|-------------|-------|
| queue | ⏳ Pending | - | - |
| scheduler | ⏳ Pending | - | - |
| Document/RAG/Graph | ⏳ Pending | - | - |

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

## Migration Checklist

For each AgenTool migration:

- [ ] Enable `use_typed_output=True` in create function
- [ ] Define or verify Output type (BaseModel)
- [ ] Update discovery operations to use success=True/False pattern
- [ ] Add field validators to Input schema
- [ ] Remove exception raising for "not found" cases
- [ ] Update tests to use typed results directly
- [ ] Remove `hasattr(result, 'output')` patterns from tests
- [ ] Verify all tests pass with type checking
- [ ] Update documentation with typed examples

## Test Pattern Migration

### Before (Untyped)
```python
result = await injector.run('tool', {...})
if hasattr(result, 'output'):
    data = json.loads(result.output)
else:
    data = result
assert data['success'] is True
```

### After (Typed)
```python
result = await injector.run('tool', {...})
assert isinstance(result, ToolOutput)
assert result.success is True
assert result.data['value'] == expected_value
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

## Notes

- Discovery operations should be idempotent and predictable
- Success=False is not an error condition, it's a valid response
- Exceptions should only be raised for genuine errors that prevent operation
- All AgenTools should follow consistent patterns for similar operations
- Type safety ensures better IDE support and runtime validation